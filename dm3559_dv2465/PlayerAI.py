import random
from BaseAI import BaseAI
from Grid import Grid
from Utils import *
import numpy as np


# Daniel Mao, Devica Verma
# Basic MiniMax with Depth limited search

LOOK_DOWN_DEPTH = 5


class PlayerAI(BaseAI):

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
        state = grid.clone()
        selfPosition = state.find(self.player_num)
        availableMoves = state.get_neighbors(selfPosition, only_available=True)

        for possibleSelfMove in availableMoves:
            state.move(possibleSelfMove, self.player_num)

            opponentPosition = state.find(3 - self.player_num)
            if len(state.get_neighbors(opponentPosition, only_available=True)) == 1:
                return possibleSelfMove

            state.move(selfPosition, self.player_num)

        bestMove = self.maximizeMove(
            grid, LOOK_DOWN_DEPTH, float('-inf'), float('inf'))[0]
        return bestMove.find(self.player_num)

    def getTrap(self, grid: Grid):
        state = grid.clone()
        enemyPosition = state.find(self.enemy_num)
        playerPosition = state.find(self.player_num)

        queue = []  # Initialize a queue
        enemyFreedomArea = 0

        queue.append(enemyPosition)

        while queue:
            s = queue.pop(0)

            for neighbour in state.get_neighbors(s, only_available=True):
                state.setCellValue(neighbour, -1)
                enemyFreedomArea = enemyFreedomArea + 1
                queue.append(neighbour)

        if self.isNeighbor(playerPosition, enemyPosition) and enemyFreedomArea == 2:
            for throw in grid.get_neighbors(grid.find(self.player_num)):
                if grid.getCellValue(throw) == -1:
                    return throw
            for throw in grid.get_neighbors(grid.find(self.enemy_num)):
                if grid.getCellValue(throw) == -1:
                    return throw

        bestTrap = self.maximizeTrap(
            grid, LOOK_DOWN_DEPTH, float('-inf'), float('inf'))[2]
        return bestTrap

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
        selfPosition = state.find(self.player_num)
        opponentPosition = state.find(3 - self.player_num)
        P = len(state.get_neighbors(selfPosition, only_available=True))
        O = len(state.get_neighbors(opponentPosition, only_available=True))

        if O >= P:
            return len(state.get_neighbors(selfPosition, only_available=True))**2 - 2*len(state.get_neighbors(opponentPosition, only_available=True))**2
        else:
            return len(state.get_neighbors(selfPosition, only_available=True))**2 - len(state.get_neighbors(opponentPosition, only_available=True))**2

    def maximizeTrap(self, state, depth, alpha, beta):
        """
        - The player tries to throw a trap.
        - Ideally the trap can be thrown at any empty cell on the grid, but
        to reduce the search space, the player tries to throw a trap in the
        enemy's neighboring cells.
        - The player is assumed static.
        """
        # find opponent
        opponent = state.find(self.enemy_num)

        # find all available cells surrounding Opponent - chance nodes
        possibleTrapThrows = state.get_neighbors(opponent, only_available=True)

        if depth <= 0 or len(possibleTrapThrows) == 0:
            return (None, self.trapHeuristic(state), state.get_neighbors(opponent)[0])

        maxChild = None
        maxUtility = float('-inf')
        bestThrow = None

        for possibleTrapThrow in possibleTrapThrows:
            if state.getCellValue(possibleTrapThrow) == self.player_num or state.getCellValue(possibleTrapThrow) == self.enemy_num:
                continue

            trapProbs = self.getThrowLikelihoods(
                state, possibleTrapThrow)
            minimizeResults = 0

            # Update board
            oldValue = state.getCellValue(possibleTrapThrow)
            state.setCellValue(possibleTrapThrow, -1)

            # Call Min (enemy)
            minimizeResults = self.minimizeTrap(
                state, depth - 1, alpha, beta)[1] * trapProbs[0]

            # Backtracking
            state.setCellValue(possibleTrapThrow, oldValue)

            # Call Min (enemy)
            if trapProbs[1] > 0:
                minimizeResults = minimizeResults + self.minimizeTrap(
                    state, depth - 1, alpha, beta)[1] * trapProbs[1]

            # Update max
            if minimizeResults > maxUtility:
                maxChild = state.clone()
                maxUtility = minimizeResults
                bestThrow = possibleTrapThrow

            # Alpha updates
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        if bestThrow is None:
            print('maximizeTrap: debug me')

        if maxChild is None:
            print('maximizeTrap: debug me')

        if maxUtility == -999:
            print('edge case detected')
            for throw in state.get_neighbors(state.find(self.player_num)):
                if state.getCellValue(throw) == -1:
                    return (maxChild, maxUtility, throw)

        return (maxChild, maxUtility, bestThrow)

    def minimizeTrap(self, state, depth, alpha, beta):
        """
        - The enemy tries to move to a valid neighboring cell.
        - The player is assumed static.
        """
        enemyPosition = state.find(self.enemy_num)

        possibleEnemyMoves = state.get_neighbors(
            enemyPosition, only_available=True)

        if depth <= 0 or len(possibleEnemyMoves) == 0:
            return (None, self.trapHeuristic(state))

        minChild = None
        minUtility = float('inf')

        for possibleEnemyMove in possibleEnemyMoves:
            # Update enemy pos
            state.move(possibleEnemyMove, self.enemy_num)

            # Call Max
            maximizeResults = self.maximizeTrap(state, depth - 1, alpha, beta)

            # Backtracking
            state.move(enemyPosition, self.enemy_num)

            # Update min
            if maximizeResults[1] < minUtility:
                minChild = state.clone()
                minUtility = maximizeResults[1]

            # Beta updates
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility

        if minChild is None:
            print('minimizeTrap: debug me')

        return (minChild, minUtility)

    def trapHeuristic(self, state):
        grid = state.clone()
        enemyPosition = state.find(self.enemy_num)

        queue = []  # Initialize a queue
        enemyFreedomArea = 0

        queue.append(enemyPosition)

        while queue:
            s = queue.pop(0)

            for neighbour in grid.get_neighbors(s, only_available=True):
                grid.setCellValue(neighbour, -1)
                enemyFreedomArea = enemyFreedomArea + 1
                queue.append(neighbour)

        return -enemyFreedomArea

    def isNeighbor(self, pos1, pos2):
        if np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1]) == 1:
            return True
        if np.abs(pos1[0] - pos2[0]) == 1 and np.abs(pos1[1] - pos2[1]) == 1:
            return True
        return False

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
