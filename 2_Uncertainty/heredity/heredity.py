import csv
import itertools
from os import name
import sys
import random
from typing import final

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # print(people)
    # print(probabilities)
    # Loop over all sets of people who might have the trait
    names = set(people)
    # print(names)
    # print(powerset(names))
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        # for person in names:
        #     print(person, people[person]["trait"], person in have_trait, have_trait, people[person]["trait"] != (person in have_trait))
        if fails_evidence:
            continue
        # print("these person have trait ")
        # print(list(have_trait))
        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    def is_Child(person):
        if person["mother"] == None and person["father"] == None:
            return False

        return True

    def get_parents(children):
        return people[children['mother']], people[children['father']]

    # returns the probability to give a gene, 
    def parent_given_gene_probability(parent):
        parent = parent['name']
        if parent in two_genes:
            return 1 - PROBS['mutation']
        elif parent in one_gene:
            return 0.5
        else:
            return PROBS['mutation']
        
    def children_given_parents_probabilities(mother_prob, father_prob, children):
        children = children['name']
        if children in two_genes:
            return mother_prob * father_prob
        elif children in one_gene:
            return mother_prob * (1 - father_prob) + (1 - mother_prob) * father_prob
        else:
            return (1 - mother_prob) * (1 - father_prob)

    # returns the probabilities of the children gene
    def parent_to_children(children):

        mother, father = get_parents(children)

        father_prob = parent_given_gene_probability(father)
        mother_prob = parent_given_gene_probability(mother)

        return children_given_parents_probabilities(mother_prob, father_prob, children)

    def parent_probs(parent):
        if parent in one_gene:
            return PROBS['gene'][1]
        elif parent in two_genes:
            return PROBS['gene'][2]
        else:
            return PROBS['gene'][0]

    def trait_probs(person):
        """ 
        calculate probability of having or not having the trait 
        """

        if person in one_gene:
            return PROBS['trait'][1][person in have_trait]
        elif person in two_genes:
            return PROBS['trait'][2][person in have_trait]
        else:
            return PROBS['trait'][0][person in have_trait]

    final_prob = 1

    for person in people:

        # different cases if is child or not
        if is_Child(people[person]):
            final_prob *= parent_to_children(people[person])
        else:
            final_prob *= parent_probs(person)

        final_prob *= trait_probs(person)

    return final_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:

        if person in one_gene:
            probabilities[person]['gene'][1] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
        else: 
            probabilities[person]['gene'][0] += p

        if person in have_trait:
            probabilities[person]['trait'][True] += p
        else:
            probabilities[person]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        genes_dict = probabilities[person]['gene']
        gene_sum = sum([genes_dict[i] for i in genes_dict])
        normalized_gene = {i: genes_dict[i] / gene_sum for i in genes_dict}
        probabilities[person]['gene'] = normalized_gene

        trait_dict = probabilities[person]['trait']
        trait_sum = sum([trait_dict[i] for i in trait_dict])
        normalized_trait = {i: trait_dict[i] / trait_sum for i in trait_dict}
        probabilities[person]['trait'] = normalized_trait


if __name__ == "__main__":
    main()
