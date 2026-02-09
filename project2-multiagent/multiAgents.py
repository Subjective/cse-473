# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        foodList = newFood.asList()

        # Reward being close to food
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1 / (minFoodDist + 1)  # prevent divide by zero

        # Avoid dangerous ghosts, chase scared ones
        # Suprisingly not much of an improvement over solely using food as a reward
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            if newScaredTimes[i] > 0:
                # Chase scared ghost
                score += 2 / (ghostDist + 1)
            elif ghostDist < 2:
                # Dangerous ghost too close: impose heavy penalty
                score -= 500

        "***You can change what this function returns***"
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        numAgents = gameState.getNumAgents()

        def value(state, depth, agentIndex):
            # Terminal state: return utility
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            # If agent is MAX (Pacman): return max-value
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex)
            # If agent is MIN (ghost): return min-value
            else:
                return minValue(state, depth, agentIndex)

        def maxValue(state, depth, agentIndex):
            v = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth - 1 if nextAgent == 0 else depth
                v = max(v, value(successor, nextDepth, nextAgent))
            return v

        def minValue(state, depth, agentIndex):
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth - 1 if nextAgent == 0 else depth
                v = min(v, value(successor, nextDepth, nextAgent))
            return v

        # Find the best action for Pacman (agent 0)
        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            nextAgent = 1 % numAgents
            nextDepth = self.depth - 1 if nextAgent == 0 else self.depth
            v = value(successor, nextDepth, nextAgent)
            if v > bestValue:
                bestValue = v
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def value(state, depth, agentIndex, alpha, beta):
            # Terminal state: return utility
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            # If agent is MAX (Pacman): return max-value
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex, alpha, beta)
            # If agent is MIN (ghost): return min-value
            else:
                return minValue(state, depth, agentIndex, alpha, beta)

        def maxValue(state, depth, agentIndex, alpha, beta):
            v = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth - 1 if nextAgent == 0 else depth
                v = max(v, value(successor, nextDepth, nextAgent, alpha, beta))
                if v > beta:  # prune
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(state, depth, agentIndex, alpha, beta):
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth - 1 if nextAgent == 0 else depth
                v = min(v, value(successor, nextDepth, nextAgent, alpha, beta))
                if v < alpha:  # prune
                    return v
                beta = min(beta, v)
            return v

        # Find the best action for Pacman (agent 0)
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            nextAgent = 1 % numAgents
            nextDepth = self.depth - 1 if nextAgent == 0 else self.depth
            v = value(successor, nextDepth, nextAgent, alpha, beta)
            if v > bestValue:
                bestValue = v
                bestAction = action
            alpha = max(alpha, v)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()

        def value(state, depth, agentIndex):
            # Terminal state: return utility
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            # If agent is MAX (Pacman): return max-value
            if agentIndex == 0:
                return maxValue(state, depth, agentIndex)
            # If agent is EXP (ghost): return exp-value
            else:
                return expValue(state, depth, agentIndex)

        def maxValue(state, depth, agentIndex):
            v = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth - 1 if nextAgent == 0 else depth
                v = max(v, value(successor, nextDepth, nextAgent))
            return v

        def expValue(state, depth, agentIndex):
            v = 0
            legalActions = state.getLegalActions(agentIndex)
            p = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth - 1 if nextAgent == 0 else depth
                v += p * value(successor, nextDepth, nextAgent)
            return v

        # Find the best action for Pacman (agent 0)
        bestAction = None
        bestValue = float("-inf")
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            nextAgent = 1 % numAgents
            nextDepth = self.depth - 1 if nextAgent == 0 else self.depth
            v = value(successor, nextDepth, nextAgent)
            if v > bestValue:
                bestValue = v
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My implementation considers the following factors and weights them by importance:
    - Game score as baseline
    - Reciprocal of closest food distance (encourages eating nearby food)
    - Penalty for remaining food count (encourages clearing the board)
    - Ghost avoidance (heavy penalty for being near dangerous ghosts)
    - Ghost hunting (reward for being near scared ghosts)
    - Penalty for remaining capsules (encourages eating power pellets)
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # Reward being close to food
    if foodList:
        minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList)
        score += 10 / (minFoodDist + 1)
    # Small penalty for remaining food (negligible compared to other factors)
    score -= 4 * len(foodList)

    # Avoid dangerous ghosts, chase scared ones
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)
        scaredTime = ghostState.scaredTimer

        if scaredTime > 0:
            # Chase scared ghost
            score += 200 / (ghostDist + 1)
        else:
            # Dangerous ghost too close: impose heavy penalty
            if ghostDist < 2:
                score -= 500

    # Capsules: slight penalty for not eating them (negligible compared to other factors)
    score -= 20 * len(capsules)

    return score


# Abbreviation
better = betterEvaluationFunction
