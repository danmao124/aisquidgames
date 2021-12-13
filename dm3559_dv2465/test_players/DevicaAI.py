import random
from BaseAI import BaseAI
from Grid import Grid
from Utils import *
import numpy as np

# Basic MiniMax with Depth limited search

LOOK_DOWN_DEPTH = 5


class DevicaAI(BaseAI):

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
        bestTrap = self.maximizeTrap(grid, LOOK_DOWN_DEPTH, float('-inf'), float('inf'))[0]
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

            # Call Min (enemy)
            minimizeResults = self.minimizeMove(state, depth - 1, alpha, beta)

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

            # update board
            oldValue = state.getCellValue(possibleTrapThrow)
            state.setCellValue(possibleTrapThrow, -1)

            # Call Max (player)
            maximizeResults = self.maximizeMove(state, depth - 1, alpha, beta)

            # Update min
            if maximizeResults[1] < minUtility:
                minChild = state.clone()
                minUtility = maximizeResults[1]

            # Backtracking
            state.setCellValue(possibleTrapThrow, oldValue)

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
        return len(state.get_neighbors(selfPosition, only_available=True)) - 2*len(state.get_neighbors(opponentPosition, only_available=True))

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
        P = self.get_player_moves(state)
        O = self.get_enemy_moves(state)
        if P >= O:
            availableMoves = state.get_neighbors(opponent, only_available=True)
            availableMoves = [neighbor for neighbor in availableMoves if state.getCellValue(neighbor) == 0]
        else:
            availableMoves = state.get_neighbors(opponent, only_available=False)
            availableMoves = [neighbor for neighbor in availableMoves if state.getCellValue(neighbor) <= 0]

        # print("#avail_moves: ", len(availableMoves))
        if depth <= 0 or len(availableMoves) == 0:
            print("Debug: ", depth, len(availableMoves), P, O)
            return (random.choice(state.getAvailableCells()), self.trapHeuristic(state))
            # return (None, self.trapHeuristic(state))

        maxChild = None
        maxUtility = float('-inf')

        for chanceNode in availableMoves:
            # Update board
            trap_positions, trap_probs = self.player_throw_trap(state, chanceNode)
            expected_utility = 0
            # Take the expected utility across all possibilities that occur when the throw
            # is intended at this chanceNode position

            # [Doubt]
            # Ideally should be across all possibilities but it is timing out, but mentioned in the PDF:
            # You will not be required to take into account the probabilities of the neighboring cells,
            # (1-p)/n, only the probability of success, p.
            for (possibleTrapThrow, trap_prob) in zip(trap_positions[:1], trap_probs[:1]):
                # Considering only the probability of intended throw position -^ [:1]
                oldValue = state.getCellValue(possibleTrapThrow)
                state.setCellValue(possibleTrapThrow, -1)

                # Call Min (enemy)
                minimizeResults = self.minimizeTrap(state, depth - 1, alpha, beta)
                expected_utility += (trap_prob * minimizeResults[1])

                # Backtracking
                state.setCellValue(possibleTrapThrow, oldValue)

            # Update max
            if expected_utility > maxUtility:
                maxChild = chanceNode
                maxUtility = expected_utility

            # Alpha updates
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility

        if maxChild is None:
            print('maximizeTrap: debug me')

        return (maxChild, maxUtility)

    def minimizeTrap(self, state, depth, alpha, beta):
        """
        - The enemy tries to move to a valid neighboring cell.
        - The player is assumed static.
        """
        enemyPosition = state.find(self.enemy_num)

        if depth <= 0:
            return (None, self.trapHeuristic(state))

        minChild = None
        minUtility = float('inf')

        possibleEnemyMoves = state.get_neighbors(
            enemyPosition, only_available=True)

        for possibleEnemyMove in possibleEnemyMoves:
            # Update enemy pos
            state.move(possibleEnemyMove, self.enemy_num)

            # Call Max
            maximizeResults = self.maximizeTrap(state, depth - 1, alpha, beta)

            # Update min
            if maximizeResults[1] < minUtility:
                minChild = state.clone()
                minUtility = maximizeResults[1]

            # Backtracking
            state.move(enemyPosition, self.enemy_num)

            # Beta updates
            if minUtility <= alpha:
                break
            if minUtility < beta:
                beta = minUtility

        if minChild is None:
            print('minimizeTrap: debug me')

        return (minChild, minUtility)

    def get_player_moves(self, state):
        selfPosition = state.find(self.player_num)
        return len(state.get_neighbors(selfPosition, only_available=True))

    def get_enemy_moves(self, state):
        enemyPosition = state.find(self.enemy_num)
        return len(state.get_neighbors(enemyPosition, only_available=True))

    def trapHeuristic(self, state):
        """
        Utility func for the trap search problem.
        """
        P = self.get_player_moves(state)
        O = self.get_enemy_moves(state)
        return  P - 2*O

    def player_throw_trap(self, grid: Grid, intended_position: tuple) -> tuple:
        '''
        Description
        ----------
        Function returns the coordinates in which the trap lands, given an intended location.

        Parameters
        ----------
        grid : current game Grid

        intended position : the (x,y) coordinates to which the player intends to throw the trap to.

        Returns
        -------
        - neighbors: Positions (x_0,y_0) in which the trap landed : list
        - probs: Probabilities of trap landing in each og the 'neighbors' positions : list
        '''

        # find neighboring cells
        neighbors = grid.get_neighbors(intended_position)

        neighbors = [neighbor for neighbor in neighbors if grid.getCellValue(neighbor) <= 0]
        n = len(neighbors)

        probs = np.ones(1 + n)

        # compute probability of success, p
        selfPosition = grid.find(self.player_num)
        p = 1 - 0.05 * (manhattan_distance(selfPosition, intended_position) - 1)

        probs[0] = p

        probs[1:] = np.ones(len(neighbors)) * ((1 - p) / n)

        # add desired coordinates to neighbors
        neighbors.insert(0, intended_position)

        return neighbors, probs