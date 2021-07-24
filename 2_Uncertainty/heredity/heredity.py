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

    # returns probability that the parent has 0 1 2 genes
    # given his trait
    def iCP(haveTrait, gene):  # AKA inverseConditionalProbability

        if gene not in PROBS['gene']:
            raise Exception("Not good gene")

        # partial sum of all the conditional probabilities
        trait = sum([PROBS['trait'][i][haveTrait] for i in PROBS['gene']])
        return PROBS['trait'][gene][haveTrait] * PROBS['gene'][gene] / trait

    # given input of genes probability calculates the probability of the trait
    def traitProbability(genes):
        return sum([PROBS['trait'][i][True] * genes[i] for i in PROBS['gene']])

    # this was for an unused func...
    def mutate(sureTrait):
        hasMutate = True if random.random() < PROBS["mutation"] else False

        if hasMutate:
            return not sureTrait
        else:
            return sureTrait

    # return the probability that parent prob
    # could gain probability from a mutation
    def mutate_with_probs(parent_prob):
        return parent_prob + (1 - parent_prob) * PROBS["mutation"] * random.random()

    def get_parents(children):
        return people[children['mother']], people[children['father']]

    def mini_normalize(tuple):
        the_sum = sum(tuple)
        return [i / the_sum for i in tuple]

    # returns the probability to give a gene, 
    def parent_given_gene_probability(parent):
        if parent['trait'] != None:
            parent_trait = parent['trait']
        else:

            # if the parent has no possible traits, just give it based on the PROBS
            return PROBS['gene'][1] * 0.5 + PROBS['gene'][2]

        # The fact the parents has 0 1 2 genes are separate
        # so i want their sum is 0.5
        parent_probs = mini_normalize((iCP(parent_trait, 0), iCP(parent_trait, 1), iCP(parent_trait, 2)))
        # print(f"the probability of the parent {parent['name']} are {parent_probs}")
        # print(f"the probability to give a gene by {parent['name']} is \n", 
        #       f"{parent_probs[1] * 0.5 + parent_probs[2]} \n", 
        #       f"and that one of not giving the gene is \n",
        #       f"{parent_probs[1] * 0.5 + parent_probs[0]} \n",
        #       f"the sum of the two probabilities is \n",
        #       parent_probs[1] * 0.5 + parent_probs[2] + parent_probs[1] * 0.5 + parent_probs[0])

        # not sure if this works, try to make some tests
        return parent_probs[1] * 0.5 + parent_probs[2]

    # returns the parent gene UNUSED, and dont know if i should make this
    def parent_given_gene(parent):
        if parent['trait'] != None:
            parent_trait = parent['trait']
            return mutate(parent_trait)

        # Else chose randomly
        return True if random.random() > 0.5 else False

    # returns the probabilities of the children gene
    def parent_to_children(children):

        mother, father = get_parents(children)

        father_prob = parent_given_gene_probability(father)
        mother_prob = parent_given_gene_probability(mother)

        father_prob = mutate_with_probs(father_prob)
        mother_prob = mutate_with_probs(mother_prob)

        zero = (1 - father_prob) * (1 - mother_prob)
        one = father_prob * (1 - mother_prob) + mother_prob * (1 - father_prob)
        two = father_prob * mother_prob

        return (zero, one, two)

    def get_children(parent):
        for person in people:
            if people[person]['mother'] == parent or people[person]['father'] == parent:
                return person

        return False

    def get_other_parent_prob(children, parent):
        other_parent = people[children]['mother'] if people[children]['mother'] != parent else people[children]['father']
        parent_trait = people[other_parent]['trait']
        return mini_normalize((iCP(parent_trait, 0), iCP(parent_trait, 1), iCP(parent_trait, 2)))

    # returns the probability of parent of having that gene (not having the traits)
    def gene_from_children(parent):
        """
        i dont think this is usefull after i have implemented this....
        i don't think i can derive the parents genes based on the children...
        its right that we can have some probabilities... but its not useful
        """
        children = get_children(parent)
        if not children:
            raise Exception("Couldn't find a children for this parent")

        children_trait = people[children]['trait']
        children_probs = mini_normalize((iCP(children_trait, 0), iCP(children_trait, 1), iCP(children_trait, 2)))
        other_parent_prob = get_other_parent_prob(children, parent)

        prob = children_probs[2] / other_parent_prob[2]  # probability of the parent to give the gene..
        return prob

    final_prob = 1

    # no_gene = [person for person in people if person not in one_gene and person not in two_genes]
    for person in people:

        person_trait = people[person]['trait']
        if person_trait != None:
            if person in one_gene:
                final_prob *= iCP(person_trait, 1)
            elif person in two_genes:
                final_prob *= iCP(person_trait, 2)
            else:
                final_prob *= iCP(person_trait, 0)

            if person in have_trait and person_trait == True:
                final_prob *= 1
            elif person in have_trait and person_trait == False:
                final_prob = 0

            continue

        if is_Child(people[person]):
            children_genes = parent_to_children(people[person])
            
            # print(f"the probabilities for the children are { children_genes } ")

            if person in one_gene:
                final_prob *= children_genes[1]
            elif person in two_genes:
                final_prob *= children_genes[2]
            else:
                final_prob *= children_genes[0]

            if person in have_trait:
                final_prob *= traitProbability(children_genes)

            continue
        
        # parent stuff calculations
        if person in one_gene:
            final_prob *= PROBS['gene'][1]
        elif person in two_genes:
            final_prob *= PROBS['gene'][2]
        else:
            final_prob *= PROBS['gene'][0]

        if person in have_trait:
            final_prob *= sum([PROBS['trait'][i][True] * PROBS['gene'][i] for i in PROBS['gene']])
        
    # print(f"and the final probability is {final_prob}")
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
