"""
Tic Tac Toe Player
"""

import random

X = "X"
O = "O"
EMPTY = None

INF = 2


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def copyBoard(board):
    newBoard = initial_state()
    length = len(board)

    for i in range(length):
        for j in range(length):
            newBoard[i][j] = board[i][j]

    return newBoard


def player(board):
    """
    The player function should take a board state as input, and return which player’s turn it is (either X or O).
    """
    numberOfXs = 0
    numberOfOs = 0

    # iteration through every column or row
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                numberOfXs += 1
            elif board[i][j] == O:
                numberOfOs += 1
    
    # just count up...
    if numberOfOs == numberOfXs:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # will append everything to this thing!
    answer = []

    # just loop through the board and look for the empty ones
    boardLength = len(board)

    for i in range(boardLength):
        for j in range(boardLength):
            if board[i][j] == EMPTY:
                answer.append((i, j))

    return answer


# Returns the board that results from making move (i, j) on the board.
def result(board, action):

    # check if its a valid action
    length = len(board)
    if action[0] >= length or action[1] >= length or action[0] < 0 or action[1] < 0 or board[action[0]][action[1]] != EMPTY:
        raise Exception

    # cant let the player do more moves if its a terminal board
    if terminal(board):
        return board

    currentPlayer = player(board)
    # print(currentPlayer, " calling from result")
    # print(action)

    newBoard = copyBoard(board)
    newBoard[action[0]][action[1]] = currentPlayer

    return newBoard


# Returns the winner of the game, if there is one.
def winner(board):
    # check rows
    for i in range(3):
        stillEqual = True
        first = board[i][0]
        for j in range(1, 3):
            if board[i][j] != first:
                stillEqual = False

        if stillEqual and first != EMPTY:
            return first

    # check columns
    for j in range(3):
        stillEqual = True
        first = board[0][j]
        for i in range(1, 3):
            if board[i][j] != first:
                stillEqual = False

        if stillEqual and first != EMPTY:
            return first

    # check diagonals
    center = board[1][1]
    if center == board[0][0] and center == board[2][2] and first != EMPTY:
        return center
    elif center == board[0][2] and center == board[2][0] and first != EMPTY:
        return center

    return None



def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) != None:
        return True

    if len(actions(board)) == 0:
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def max_giocata(board, alpha=INF):
    """
    returns max value possible for a particular board
    """
    # se ho finito ritorno il risultato corrente
    if terminal(board):
        return utility(board)

    # solamente un numero piu piccolo di tutti quelli possibili
    v = -INF
    moves = actions(board)

    for move in moves:
        v = max(v, min_giocata(result(board, move), v))
        if v > alpha:
            return v

    return v


def min_giocata(board, beta=-INF):
    """
    returns min value possible for a particular board
    """
    if terminal(board):
        return utility(board)

    # solamente un numero piu grande di tutti quelli possibili
    v = INF

    moves = actions(board)

    for move in moves:
        v = min(v, max_giocata(result(board, move), v))
        if v < beta:
            return v

    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board
    """
    if terminal(board):
        return None

    moves = actions(board)
    totalMoves = len(moves)

    if player(board) == X:
        v = -INF
        movesResult = []
        for i in range(totalMoves):
            next = min_giocata(result(board, moves[i]))
            movesResult.append(next)
            # print(f"value of the move {moves[i]} is {next}, current v: {v}")
        
        maxValue = max(movesResult)

        maxIndexes = [i for i in range(totalMoves) if movesResult[i] == maxValue]

        returnIndex = maxIndexes[0]  # maxIndexes[random.randrange(0, len(maxIndexes))]

        # print(moves[returnIndex], " from x player: ", maxValue)
        return moves[returnIndex]    
    else:
        v = INF
        movesResult = []
        for i in range(totalMoves):
            next = max_giocata(result(board, moves[i]))
            movesResult.append(next)
            # print(f"value of the move {moves[i]} is {next}, current v: {v}")

        minValue = min(movesResult)
        minIndexes = [i for i in range(totalMoves) if movesResult[i] == minValue]
        returnIndex = minIndexes[0]  # minIndexes[random.randrange(0, len(minIndexes))]

        # print(moves[returnIndex], " from O player: ", minValue)
        return moves[returnIndex] 


# some debuging attemps
if __name__ == "__main__":
    pass
    # result(initial_state(), (3,3))

    ########
    # b = initial_state()
    # b[0][0] = X
    # b[0][1] = O
    # b[0][INF] = X
    # print(player(b))