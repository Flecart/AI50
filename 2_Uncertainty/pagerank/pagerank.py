import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    answer = dict()
    links_in_page = len(corpus[page])
    for key in corpus:
        if key not in corpus[page]:
            answer[key] = (1 - damping_factor) / len(corpus)
    for key in corpus[page]:
        # 1 sarebbe la probability di essere sulla pagina corrente
        answer[key] = (1 - damping_factor) / len(corpus) + damping_factor * 1 / links_in_page 

    return answer


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    def get_random_page(thresholds, value):

        # sanity 
        if value > thresholds[-1][0]:
            raise Exception("strange error")

        i = 1
        while value > thresholds[i][0]:
            i += 1

        return thresholds[i][1]

    def make_threshholds(page):
        thresholds = [(0, 0)]
        probabilities = transition_model(corpus, page, damping_factor)
        # print(f"these are the probabilities {probabilities}, at {page}")
        i = 0
        for key in probabilities:
            # make thresholds for random values later
            thresholds.append((thresholds[i][0] + probabilities[key], key))
            i += 1

        return thresholds

    answer = dict()
    
    first_page = random.randrange(0, len(corpus))
    random_chosen_page = [key for key in corpus][first_page]
    # initialize values for answer dict
    for key in corpus:
        answer[key] = 0
        if key == random_chosen_page:
            answer[key] = 1
    
    thresholds = make_threshholds(random_chosen_page)

    for _ in range(n - 1):
        random_value = random.random() * thresholds[-1][0]  # that is the max value
        key = get_random_page(thresholds, random_value)
        answer[key] += 1
        thresholds = make_threshholds(key)
        # print(thresholds)

    for key in answer:
        answer[key] = answer[key] / n

    # print(answer, sum([answer[key] for key in answer]))
    # assert (sum([answer[key] for key in answer]) == 1) # python floating point imprecision....

    return answer


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # update the changes array
    def update_changes(old_sample, new_sample):
        i = 0
        for key in old_sample:
            changes[i] = abs(old_sample[key] - new_sample[key])
            i += 1

    # see if the answer dics has changed enough
    def has_changed_enough():
        for number in changes:
            if number > LIMIT:
                return True
        return False

    def normalize(dictionary):
        current_sum = sum([dictionary[key] for key in dictionary])
        return {key: dictionary[key] / current_sum for key in dictionary}

    def invert(corpus_email):
        my_dict = {key: set() for key in corpus_email}

        for key in corpus_email:
            for connection in corpus_email[key]:
                my_dict[connection].add(key)

        return my_dict

    LIMIT = 0.0005

    corpus_len = len(corpus)
    answer = {key: 1 / corpus_len for key in corpus}
    changes = [LIMIT + 1] * corpus_len

    inverted_corpus = invert(corpus)
    while has_changed_enough():
        new = {key: ((1 - damping_factor) / corpus_len + damping_factor * 
                     sum([answer[connections] / len(corpus[connections]) for connections in inverted_corpus[key]])) for key in answer}
        new = normalize(new)
        update_changes(new, answer)
        answer = new

    # print(answer, sum([answer[key] for key in answer]))
    # assert (sum([answer[key] for key in answer]) == 1)
    return answer


if __name__ == "__main__":
    main()