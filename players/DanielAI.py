import random
from BaseAI import BaseAI
import numpy as np
from Grid import Grid
from Utils import *
import itertools

# Daniel Mao
# Basic MiniMax with Depth limited search


class DanielAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()

    def getMove(self, grid):
        """ Returns a random, valid move """

        bestMove = self.maximizeMove(grid, 3)[0]
        newPos = np.where(bestMove.map == 1)
        newPos = (newPos[0][0], newPos[1][0])

        self.setPosition(newPos)

        return newPos

    def getTrap(self, grid: Grid):
        """Get the *intended* trap move of the player"""

        # find all available cells in the grid
        available_cells = grid.getAvailableCells()

        # find all available cells
        trap = random.choice(available_cells) if available_cells else None

        return trap

    def setPosition(self, new_pos: tuple):
        self.pos = new_pos

    def getPosition(self):
        return self.pos

    # The algorithm for figuring out the safest place to move. State is the
    # current board, and depth is the max search depth. Once it reaches 0, we stop.
    def maximizeMove(self, state, depth):
        selfPosition = np.where(state.map == 1)
        selfPosition = (selfPosition[0][0], selfPosition[1][0])
        availableMoves = state.get_neighbors(selfPosition, only_available=True)

        if depth <= 0 or len(availableMoves) == 0:
            return (None, self.moveHeuristic(state, selfPosition))

        maxChild = None
        maxUtility = float('-inf')

        for possibleSelfMove in availableMoves:
            state.move(possibleSelfMove, 1)

            enemyPosition = np.where(state.map == 2)
            enemyPosition = (enemyPosition[0][0], enemyPosition[1][0])
            possibleEnemyTrapLocations = state.get_neighbors(
                enemyPosition, only_available=False)

            for possibleEnemyTrapLocation in possibleEnemyTrapLocations:
                if state.getCellValue(possibleEnemyTrapLocation) == 1 or state.getCellValue(possibleEnemyTrapLocation) == 2:
                    continue

                # update board
                oldValue = state.getCellValue(possibleEnemyTrapLocation)
                state.setCellValue(possibleEnemyTrapLocation, -1)

                # Call Min (enemy)
                minimizeResults = self.minimizeMove(state, depth - 1)

                # Update max
                if minimizeResults[1] > maxUtility:
                    maxChild = state.clone()
                    maxUtility = minimizeResults[1]

                # Backtracking
                state.setCellValue(possibleEnemyTrapLocation, oldValue)
            state.move(selfPosition, 1)

        if maxChild is None:
            print('debug me')

        return (maxChild, maxUtility)

    def minimizeMove(self, state, depth):
        selfPosition = np.where(state.map == 2)
        selfPosition = (selfPosition[0][0], selfPosition[1][0])
        availableMoves = state.get_neighbors(selfPosition, only_available=True)

        if depth <= 0 or len(availableMoves) == 0:
            return (None, self.moveHeuristic(state, selfPosition))

        minChild = None
        minUtility = float('inf')

        for possibleSelfMove in state.get_neighbors(selfPosition, only_available=True):
            state.move(possibleSelfMove, 2)

            enemyPosition = np.where(state.map == 1)
            enemyPosition = (enemyPosition[0][0], enemyPosition[1][0])
            possibleEnemyTrapLocations = state.get_neighbors(
                enemyPosition, only_available=False)

            for possibleEnemyTrapLocation in possibleEnemyTrapLocations:
                if state.getCellValue(possibleEnemyTrapLocation) == 1 or state.getCellValue(possibleEnemyTrapLocation) == 2:
                    continue

                # update board
                oldValue = state.getCellValue(possibleEnemyTrapLocation)
                state.setCellValue(possibleEnemyTrapLocation, -1)

                # Call Max (enemy)
                maximizeResults = self.maximizeMove(state, depth - 1)

                # Update min
                if maximizeResults[1] < minUtility:
                    minChild = state.clone()
                    minUtility = maximizeResults[1]

                # Backtracking
                state.setCellValue(possibleEnemyTrapLocation, oldValue)
            state.move(selfPosition, 2)

        if minChild is None:
            print('debug me')

        return (minChild, minUtility)

    def moveHeuristic(self, state, selfPosition):
        return len(state.get_neighbors(selfPosition, only_available=True))
