from types import resolve_bases
import nltk
import sys
import os
import string 
from math import log

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    ans = dict()
    for f in os.listdir(directory):
        filename, file_extension = os.path.splitext(f)
        if file_extension == ".txt":
            with open(os.path.join(directory, f), 'r') as content:

                # error if there is file with the same name
                filename += file_extension
                if filename in ans:
                    raise Exception(f"Can't load file { filename } because of duplicate ")

                ans[filename] = content.read()

    return ans


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # getting rid of all the punctuation in the document
    # found here https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    document = document.translate(str.maketrans('', '', string.punctuation))

    # returns true if the word has punctuation
    def isPuntuation(word):

        # i just want to filter words with len = 1
        if len(word) == 1:
            return True 

        for char in word:
            if char in string.punctuation:
                return True

        return False

    # getting only the words not in stopwords
    words = [ word.lower() for word in nltk.word_tokenize(document) if word.lower() not in nltk.corpus.stopwords.words("english")]

    # debugging to see what it has filtered
    # print(words)    
    # raise
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    
    total_documents = len(documents.keys())
    for doc in documents.values():
        # looping through all words in a doc
        for token in doc:

            # don't want to look for already counted words           
            if token in idfs:
                continue

            # counting the occurences 
            # similiar to script here https://cdn.cs50.net/ai/2020/spring/lectures/6/src6/tfidf/tfidf.py
            count = sum(token in documents[filename] for filename in documents)
            idfs[token] = log(total_documents / count)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # maps filename to tf_idfs value
    tf_idfs = {filename: 0 for filename in files}

    for word in query:
        for filename, content in files.items():
            term_frequency = content.count(word)
            tf_idfs[filename] += term_frequency * idfs[word]

    # extracting the filename in a sorted tuple list
    return_list = [name for name, value in sorted(tf_idfs.items(), key=lambda item: item[1], reverse=True)]

    return return_list[0:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # getting the sentences_idfs and term density
    # i need a set because in this way i don't get duplicates(i don't count multiple times)
    sentence_idfs = {
        key: ({word for word in sentences[key] if word in query}, 
        sum([word in query for word in sentences[key]]) / len(sentences[key])) 
        for key in sentences
    }

    # from set to float sum
    sentence_idfs = {
        key: (sum([idfs[word] for word in value[0]]),
        value[1])
        for key, value in sentence_idfs.items()
    }
    
    # last sort, should change only in case of ties
    # see here: https://stackoverflow.com/questions/50885410/python-sort-dictionary-by-descending-values-and-then-by-keys-alphabetically/50885442
    # i'm sorting with the second tuple.
    return_list = sorted(sentence_idfs, key=lambda x: sentence_idfs[x], reverse=True)

    # Following code was used to debug the python query in the example in
    # https://cs50.harvard.edu/ai/2020/projects/6/questions/
        # print(sentences['Python 3.0, released in 2008, was a major revision of the language that is not completely backward-compatible, and much Python 2 code does not run unmodified on Python 3.'])
        # print(query)
        # print([(name, value) for name, value in sorted(sentence_idfs.items(), key=lambda item: item[1][0], reverse=True)][0:5])
        
    return return_list[0:n]


if __name__ == "__main__":
    main()
