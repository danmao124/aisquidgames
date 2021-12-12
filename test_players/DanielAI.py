import random
from BaseAI import BaseAI
from Grid import Grid
from Utils import *
import numpy as np


# Daniel Mao
# Basic MiniMax with Depth limited search

LOOK_DOWN_DEPTH = 5


class DanielAI(BaseAI):

    def __init__(self, initial_position=None) -> None:
        super().__init__()
        self.pos = initial_position
        self.player_num = None
        self.enemy_num = None

    def setPosition(self, new_pos: tuple):
        self.pos = new_pos

    def getPosition(self):
        return self.pos

    def setPlayerNum(self, num):
        self.player_num = num
        self.enemy_num = abs(num - 3)

    def getMove(self, grid):
        """ Returns a random, valid move """
        bestMove = self.maximizeMove(
            grid, LOOK_DOWN_DEPTH, float('-inf'), float('inf'))[0]
        return bestMove.find(self.player_num)

    def getTrap(self, grid: Grid):
        # find players
        opponent = grid.find(3 - self.player_num)

        # find all available cells in the grid
        available_neighbors = grid.get_neighbors(opponent, only_available=True)

        # edge case - if there are no available cell around opponent, then
        # player constitutes last trap and will win. throwing randomly.
        if not available_neighbors:
            return random.choice(grid.getAvailableCells())

        states = [grid.clone().trap(cell) for cell in available_neighbors]

        # find trap that minimizes opponent's moves
        is_scores = np.array([IS(state, 3 - self.player_num)
                              for state in states])

        # throw to one of the available cells randomly
        trap = available_neighbors[np.argmin(is_scores)]

        return trap

    # The algorithm for figuring out the safest place to move. State is the
    # current board, and depth is the max search depth. Once it reaches 0, we stop.

    def maximizeMove(self, state, depth, alpha, beta):
        selfPosition = state.find(self.player_num)
        availableMoves = state.get_neighbors(selfPosition, only_available=True)

        if depth <= 0 or len(availableMoves) == 0:
            return (None, self.moveHeuristic(state))

        maxChild = None
        maxUtility = float('-inf')

        for possibleSelfMove in availableMoves:
            state.move(possibleSelfMove, self.player_num)

            minimizeResults = self.minimizeMove(
                state, depth - 1, alpha, beta)

            # Update max
            if minimizeResults[1] > maxUtility:
                maxChild = state.clone()
                maxUtility = minimizeResults[1]

            # Backtracking
            state.move(selfPosition, self.player_num)

            # Alpha updates
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        if maxChild is None:
            print('debug me')

        return (maxChild, maxUtility)

    def minimizeMove(self, state, depth, alpha, beta):
        enemyPosition = state.find(self.player_num)

        if depth <= 0:
            return (None, self.moveHeuristic(state))

        minChild = None
        minUtility = float('inf')

        possibleTrapThrows = state.get_neighbors(
            enemyPosition, only_available=False)

        for possibleTrapThrow in possibleTrapThrows:
            if state.getCellValue(possibleTrapThrow) == self.player_num or state.getCellValue(possibleTrapThrow) == self.enemy_num:
                continue

            trapProbs = self.getThrowLikelihoods(
                state, possibleTrapThrow)
            maximizeResults = 0

            # update board
            oldValue = state.getCellValue(possibleTrapThrow)
            state.setCellValue(possibleTrapThrow, -1)

            # Call Max (player)
            maximizeResults = self.maximizeMove(
                state, depth - 1, alpha, beta)[1] * trapProbs[0]

            # Backtracking
            state.setCellValue(possibleTrapThrow, oldValue)

            # Call Max (player)
            if trapProbs[1] > 0:
                maximizeResults = maximizeResults + self.maximizeMove(
                    state, depth - 1, alpha, beta)[1] * trapProbs[1]

            # Update min
            if maximizeResults < minUtility:
                minChild = state.clone()
                minUtility = maximizeResults

            # Beta updates
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility

        if minChild is None:
            print('debug me')

        return (minChild, minUtility)

    def moveHeuristic(self, state):
        # 48/50 vs medium bot
        selfPosition = state.find(self.player_num)
        opponentPosition = state.find(3 - self.player_num)
        return len(state.get_neighbors(selfPosition, only_available=True))**2 - 2*len(state.get_neighbors(opponentPosition, only_available=True))**2

        # 41/50 vs medium bot
        # selfPosition = state.find(self.player_num)
        # opponentPosition = state.find(3 - self.player_num)
        # return len(state.get_neighbors(selfPosition, only_available=True)) - 2*len(state.get_neighbors(opponentPosition, only_available=True))

        # 42/50 vs medium bot
        # selfPosition = state.find(self.player_num)
        # opponentPosition = state.find(3 - self.player_num)
        # P = len(state.get_neighbors(selfPosition, only_available=True))
        # O = len(state.get_neighbors(opponentPosition, only_available=True))
        # if (2*P - O) >= (P - 2*O):
        #     return P - 2*O
        # else:
        #     return P - O

        # 46/50 vs medium bot
        # selfPosition = state.find(self.player_num)
        # opponentPosition = state.find(3 - self.player_num)
        # P = len(state.get_neighbors(selfPosition, only_available=True))
        # O = len(state.get_neighbors(opponentPosition, only_available=True))
        # if (2*P - O) >= (P - 2*O):
        #     return len(state.get_neighbors(selfPosition, only_available=True))**2 - 2*len(state.get_neighbors(opponentPosition, only_available=True))**2
        # else:
        #     return len(state.get_neighbors(selfPosition, only_available=True))**2 - len(state.get_neighbors(opponentPosition, only_available=True))**2

        # 44/50 vs medium bot
        # selfPosition = state.find(self.player_num)
        # opponentPosition = state.find(3 - self.player_num)
        # P = len(state.get_neighbors(selfPosition, only_available=True))
        # O = len(state.get_neighbors(opponentPosition, only_available=True))
        # if (2*P - O) >= (P - 2*O):
        #     return len(state.get_neighbors(selfPosition, only_available=True)) - len(state.get_neighbors(opponentPosition, only_available=True))**2
        # else:
        #     return len(state.get_neighbors(selfPosition, only_available=True))**2 - len(state.get_neighbors(opponentPosition, only_available=True))

    def getThrowLikelihoods(self, grid: Grid, intended_position: tuple) -> tuple:
        '''
        Description
        ----------
        Function returns the probability that the trap lands in the intended position, or lands in a spot causing no change.

        Parameters
        ----------
        grid : current game Grid

        intended position : the (x,y) coordinates to which the player intends to throw the trap to.

        Returns
        -------
        - probs: (probability of landing in intended position, probability of causing no change) : list
        '''

        # find neighboring cells
        originalNeighbors = grid.get_neighbors(intended_position)

        neighbors = [
            neighbor for neighbor in originalNeighbors if grid.getCellValue(neighbor) <= 0]
        n = len(neighbors)

        probs = np.ones(2)

        # compute probability of success, p
        selfPosition = grid.find(self.player_num)
        p = 1 - 0.05 * \
            (manhattan_distance(selfPosition, intended_position) - 1)

        probs[0] = p
        probs[1] = len([
            neighbor for neighbor in originalNeighbors if grid.getCellValue(neighbor) != 0]) * ((1 - p) / n)

        return probs


def IS(grid: Grid, player_num):
            # find all available moves by Player
    player_moves = grid.get_neighbors(
        grid.find(player_num), only_available=True)

    # find all available moves by Opponent
    opp_moves = grid.get_neighbors(
        grid.find(3 - player_num), only_available=True)

    return len(player_moves) - len(opp_moves)
