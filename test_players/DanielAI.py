import random
from BaseAI import BaseAI
from Grid import Grid
from Utils import *

# Daniel Mao
# Basic MiniMax with Depth limited search

LOOK_DOWN_DEPTH = 3


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

        bestMove = self.maximizeMove(grid, LOOK_DOWN_DEPTH)[0]
        return bestMove.find(self.player_num)

    def getTrap(self, grid: Grid):
        # find opponent
        opponent = grid.find(3 - self.player_num)

        # find all available cells surrounding Opponent
        available_cells = grid.get_neighbors(opponent, only_available=True)

        # throw to one of the available cells randomly
        trap = random.choice(available_cells)

        return trap

    # The algorithm for figuring out the safest place to move. State is the
    # current board, and depth is the max search depth. Once it reaches 0, we stop.
    def maximizeMove(self, state, depth):
        selfPosition = state.find(self.player_num)
        availableMoves = state.get_neighbors(selfPosition, only_available=True)

        if depth <= 0 or len(availableMoves) == 0:
            return (None, self.moveHeuristic(state))

        maxChild = None
        maxUtility = float('-inf')

        possibleEnemyTrapThrows = state.get_neighbors(
            selfPosition, only_available=False)

        for possibleEnemyTrapThrow in possibleEnemyTrapThrows:
            if state.getCellValue(possibleEnemyTrapThrow) == self.player_num or state.getCellValue(possibleEnemyTrapThrow) == self.enemy_num:
                continue

            # update board because we need to get available player moves
            oldValue = state.getCellValue(possibleEnemyTrapThrow)
            state.setCellValue(possibleEnemyTrapThrow, -1)

            availableMoves = state.get_neighbors(
                selfPosition, only_available=True)
            for possibleSelfMove in availableMoves:
                state.move(possibleSelfMove, self.player_num)

                # Call Min (enemy)
                minimizeResults = self.minimizeMove(state, depth - 1)

                # Update max
                if minimizeResults[1] > maxUtility:
                    maxChild = state.clone()
                    maxUtility = minimizeResults[1]

                # Backtracking
                state.move(selfPosition, self.player_num)
            state.setCellValue(possibleEnemyTrapThrow, oldValue)

        if maxChild is None:
            print('debug me')

        return (maxChild, maxUtility)

    def minimizeMove(self, state, depth):
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
            maximizeResults = self.maximizeMove(state, depth - 1)

            # Update min
            if maximizeResults[1] < minUtility:
                minChild = state.clone()
                minUtility = maximizeResults[1]

            # Backtracking
            state.setCellValue(possibleTrapThrow, oldValue)

        if minChild is None:
            print('debug me')

        return (minChild, minUtility)

    def moveHeuristic(self, state):
        selfPosition = state.find(self.player_num)
        return len(state.get_neighbors(selfPosition, only_available=True))
