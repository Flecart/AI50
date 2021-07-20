import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells.copy()

        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells.copy()

        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        # check if in cell
        if cell not in self.cells:
            return

        # removing the marked cell
        self.cells.remove(cell)

        # updating the count 
        self.count -= 1

        return

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        # check if in cell
        if cell not in self.cells:
            return

        # just remove the cell
        self.cells.remove(cell)

        # safety check, later i discovered this is useless
        # if len(self.cells) > self.count:
        #     raise Exception("mmm somethign bad has happended, very bad")

        return


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.
        """
        # mark the cell as a move that has been made
        self.moves_made.add(cell)

        # mark the cell as safe
        self.mark_safe(cell)

        # add a new sentence to the AI's knowledge base base on the value of `cell` and `count` 
        # creating set of neighbours
        neighbours = set()

        for i in range(max(0, cell[0] - 1), min(self.width, cell[0] + 2)):
            for j in range(max(0, cell[1] - 1), min(self.height, cell[1] + 2)):
                # print(f"looking at {i, j} now")
                # dont add known mine?
                if (i, j) in self.mines:
                    count -= 1
                    continue

                # be sure not having already explored this neighbour or safe one
                if (i, j) not in self.moves_made and not (i, j) in self.safes:
                    # print(f"adding couples {i, j} to the sentence with count {count}")
                    neighbours.add((i, j))

        new_sentence = Sentence(neighbours, count)
        self.knowledge.append(new_sentence)

        # Mark any additional cells as safe or as mines if it can be concluded base on the AI's s knowledge base
        
        for sentence in self.knowledge:
            safes = sentence.known_safes()
            for temp_cell in safes:
                self.mark_safe(temp_cell)
            mines = sentence.known_mines()
            for temp_cell in mines:
                self.mark_mine(temp_cell)

        # Add any new sentences to the AI's knowledge base if they can be inferred from existing knowledge
        
        old_knowledge = self.knowledge.copy()[:-1]  # dont take new sentence!
        assert(self.knowledge[-1] == new_sentence)
        sum = 0
        for old_sentence in old_knowledge:
            sum += 1
            if len(old_sentence.cells) > 0 and old_sentence.cells.issubset(new_sentence.cells):
                difference_set = old_sentence.cells - new_sentence.cells
                diffrence_count = old_sentence.count - new_sentence.count
                # (i know its at the last point!) i have to remove this bc the bigger one has no more info than the new one!
                del self.knowledge[-1] 
                self.knowledge.append(Sentence(difference_set, diffrence_count))
            elif len(new_sentence.cells) > 0 and new_sentence.cells.issubset(old_sentence.cells):
                difference_set = old_sentence.cells - new_sentence.cells
                diffrence_count = old_sentence.count - new_sentence.count
                self.knowledge.remove(old_sentence)  # same as before!
                self.knowledge.append(Sentence(difference_set, diffrence_count))

        # print(f"watched {sum} couples!")
        
        

        # REGION another very memory heavy algorithm for making inferences!

        # knowledge_length = len(self.knowledge)
        # tmp_set = []
        
        # i discovred i should watch every possible combination, but just with the noew one!
        # for i in range(knowledge_length):
        #     first = self.knowledge[i]
        #     sum = 0
        #     for j in range(knowledge_length):
        #         # print(f"i' m currently comparing {i, j} knowledge stuff")
        #         sum += 1
        #         second = self.knowledge[j]

        #         if first.cells.issubset(second.cells):
        #             difference_set = second.cells - first.cells
        #             difference_count = second.count - first.count
        #             tmp_set.append(Sentence(difference_set, difference_count))
        #         elif second.cells.issubset(first.cells):
        #             difference_set = first.cells - second.cells
        #             difference_count = first.count - second.count
        #             tmp_set.append(Sentence(difference_set, difference_count))

        #     print(f"i just saw {sum} couples")
        # # putting the stuff i added back in main set
        # self.knowledge += tmp_set

        # REGION very memory heavy inference check
        # we will pick one from one muahaha, and see if can make an inference leeel
        # combinations = list(itertools.combinations(self.knowledge, 2))
        # for i in range(len(combinations)):
        #     first = combinations[i][0]
        #     second = combinations[i][1]
        #     # checking if we can find some subsets
        #     if first.cells.issubset(second.cells):
        #         difference_set = second.cells - first.cells
        #         difference_count = second.count - first.count
        #         self.knowledge.append(Sentence(difference_set, difference_count))
        #     elif second.cells.issubset(first.cells):
        #         difference_set = first.cells - second.cells
        #         difference_count = first.count - second.count
        #         self.knowledge.append(Sentence(difference_set, difference_count))
        # ENDREGION

        return

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for move in self.safes:
            if move not in self.moves_made:
                return move

        # if cant find any before just return none
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # check if there are still possible moves
        if len(self.moves_made) + len(self.mines) >= self.width * self.height:
            return None

        move = (random.randrange(0, self.width), random.randrange(0, self.height))  # just initialize this
        while move in self.moves_made or move in self.mines:
            move = (random.randrange(0, self.width), random.randrange(0, self.height))

        # if it finds a good move exits and returns a move, hoping its fine
        return move
                
