import sys
from queue import SimpleQueue as Queue
from types import resolve_bases

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            length = var.length
            to_be_removed = []

            for word in self.domains[var]:
                if len(word) != length:
                    to_be_removed.append(word)

            for word in to_be_removed:
                self.domains[var].remove(word)     

    # Implemented with the help of the pseudoalgorithm in 
    # https://cs50.harvard.edu/ai/2020/notes/3/
    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """

        # filter overlapped
        if self.crossword.overlaps[(x, y)] == None:
            return False
        else:
            xIndex, yIndex = self.crossword.overlaps[(x, y)]

        to_be_deleted = []

        # checking values in X.domain
        for xWord in self.domains[x]:
            satisfied = False
            for yWord in self.domains[y]:
                if yWord[yIndex] == xWord[xIndex]:
                    satisfied = True
                    break

            if not satisfied:
                to_be_deleted.append(xWord)

        if len(to_be_deleted) != 0:
            for word in to_be_deleted:
                self.domains[x].remove(word)     
            return True

        return False
    
    # Implemented with the help of the pseudoalgorithm in 
    # https://cs50.harvard.edu/ai/2020/notes/3/
    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = Queue()
        if arcs == None:
            arcs = [key for key in self.crossword.overlaps]
        for arc in arcs:
            queue.put(arc)

        while not queue.empty():
            x, y = queue.get()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                neighbors = self.crossword.neighbors(x)
                neighbors.remove(y)
                for z in neighbors:
                    queue.put((z,x))

        
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for key in self.domains:
            # if a var is not present then its not complete
            if key not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        overlaps = self.crossword.overlaps

        # checking all possible characters overlaps
        for overlap in overlaps:
            x, y = overlap
            intersection = overlaps[overlap]

            # checking only overlaps present in the assignment
            if x in assignment and y in assignment and intersection != None:
                xInters, yInters = intersection

                # conflicting characters detected
                # print(xInters)
                # print(assignment[x])
                # print(yInters)
                # print(assignment[y])
                if assignment[x][xInters] != assignment[y][yInters]:
                    return False
        
        # checking words used twice or more
        used_words = []
        for key in assignment:
            if not assignment[key] in used_words:
                used_words.append(assignment[key])
            else:
                return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        # creating a list of valid neighbors
        neighbors = {key[1]: value for key, value in self.crossword.overlaps.items() if key[0] == var and value != None and key[1] not in assignment} 
        values = {}

        # making choices for values of current variable
        for value in self.domains[var]:
            values[value] = 0

            # check how much this choice is constraining neighbors
            for neighbor in neighbors:
                xInter, yInter = neighbors[neighbor]
                for neigh_values in self.domains[neighbor]:
                    # constrain detected
                    if neigh_values[yInter] != value[xInter]:
                        values[value] += 1

        # sorting as specified in https://docs.python.org/3/howto/sorting.html
        return sorted(values.keys(), key=values.__getitem__)

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # dictionary of tuples, number of remaining (values and degree)
        not_used = {}
        for var in self.domains:
            if var not in assignment:
                not_used[var] = [0, len(self.crossword.neighbors(var))]

        #  checking possible values for each variable
        for var in not_used:
            # new entry to check consistency
            assignment[var] = 0

            for values in self.domains[var]:
                assignment[var] = values
                if self.consistent(assignment):
                    not_used[var][0] += 1

            assignment.pop(var)

        not_used = sorted(not_used.items(), key=lambda x: x[1][0])
        same_values = []

        # minimum number of remaining values in its domain
        min = not_used[0][1][0]  
        for entries in not_used:
            if min == entries[1][0]:
                same_values.append(entries)

        same_values = sorted(same_values, key=lambda x: x[1][1], reverse=True)
        return same_values[0][0]
                
        

    # Implemented with the help of the pseudoalgorithm in 
    # https://cs50.harvard.edu/ai/2020/notes/3/
    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        # print(assignment)
        # print(self.domains)
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result != None:
                    return result
                assignment.pop(var)
            else:
                assignment.pop(var)
                
        return None



def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()
    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
