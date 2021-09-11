import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# done with the initial help of the notes here:
# https://cdn.cs50.net/ai/2020/spring/lectures/6/src6/cfg/cfg1.py
NONTERMINALS = """
S -> Phrase | Phrase Conj Phrase | Phrase Conj VP
Phrase -> NP VP

AP -> Adj | Adj AP
NP -> N | Det N
NP ->  PP NP | AP NP | NP PP | Det AP N

VP -> V | V NP | V PP
VP -> AdvP V | AdvP V NP | AdvP V NP PP | AdvP V PP
VP ->  V AdvP |  V AdvP NP |  V AdvP NP PP |  V AdvP PP

PP -> P NP | P NP AdvP
AdvP -> Adv | Adv AdvP
"""
# VP -> | V NP PP 
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # python string comprehension
    # keeping only words that have chars or numbers but are not only digits.
    words = [word.lower() for word in nltk.word_tokenize(sentence) if not word.isdigit() and word.isalnum()]
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    ans = list()
    for i in range(3, tree.height()):

        for s in tree.subtrees(lambda t: t.height() == i):
            tmp = str(s)

            # it will return something like
            # '(NP the holmes)'

            # "(NP" has to occur once at the beginning
            if tmp.count("(NP") == 1 and tmp[0:3] == "(NP":
                ans.append(s)
    return ans


if __name__ == "__main__":
    main()
