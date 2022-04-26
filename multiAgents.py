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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # Begin your code (Part 1)
        """
        1. Use a recursion to count for the result.
        2. The gameState is used to pass into the "scoreEvaluationFunction" and get the "score" back.
        3. The pacman (agent_index = 0) want the next step to be the "max" score - max player
           The ghost (agent_index != 0) want the next step to be the "min" score - min player
        4. For the score of "next step" we need to use the "next gameState" to get the value
        """
        action, score = self.minimax( 0, 0, gameState ) # Get the action of pacman (agent_index = 0)
        return action

    def minimax( self, cur_depth, agent_index, gameState ):
        """
        1. Get all the actions the current agents will take to be the potential next actions 
            by using "gameState.getLegalActions()"

        2. Use the "potential next actions" & "next agent index" to get the "next gameState" 
            by using "gameState.getNextState(agentIndex, action)"

        3. Use the "next gameState" to count for the score, and keep recording to a score list
        
        4. The evalutionFinction( gameState ) can only be used for the terminal gameState,
            which means there will only be score for the "terminal state".
            We need to count out the middle points score by "finding back" from "terminal state score".
            -> this is why we need recursion for the answer for each layer
            -> or we actually need the "terminal value" in "minimax algorithm", so it is forbidden to use
                the middle value in this algorithm by passing the current state into the "evalutionFunction"

        5. Extract the min/max_score from the score list and return it back with it's corresponding best_action
        
        6. If the game is over, the action_list will be empty, the score_list will also be empty,
           it will be illegal to do step5, thus we need to return this situation at first.

        7. If agent_index > gameState.getNumAgents -> we have finish a depth
           roll back the agent_index to 0, and add 1 to the cur_depth

        8. if cur_depth == self.depth, we have already finish all depth, which is 0 ~ self.depth-1,
           also return the value as self.evaluationFunction( gameState )
        """
        if gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction( gameState )   # return with score == inf/-inf

        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            cur_depth += 1

        if cur_depth == self.depth:
            return None, self.evaluationFunction( gameState )

        action_list = gameState.getLegalActions(agent_index)
        score_list = []
        for action in action_list:
            next_state = gameState.getNextState( agent_index, action )  # the value need to be came from the bottom
            _, score = self.minimax( cur_depth, agent_index+1, next_state )
            score_list.append( score )

        if agent_index == 0:
            return action_list[ score_list.index( max(score_list) ) ], max(score_list)
        else:
            return action_list[ score_list.index( min(score_list) ) ], min(score_list)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Begin your code (Part 2)
        action, score = self.alphaBeta( 0, 0, gameState, float("-inf"), float("inf") )
        return action

    def alphaBeta( self, cur_depth, agent_index, gameState, alpha, beta ):
        """
        *     The code structure refer to the sudo code on wiki      *
        * https://zh.wikipedia.org/wiki/Alpha-beta%E5%89%AA%E6%9E%9D *

        1. The best value of child will update the value of parent,
            by returning its "own" value to parent, but "not its alpha, beta value"
            -> only effect "its own root" trait

        2. if alpha > beta -> pruning 

        3. max layer: alpha = max( alpha, score )
            min layer: beta  = min( beta,  score )

        4. keep recording for the best_score and best_action
        """
        if gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction( gameState )

        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            cur_depth += 1

        if cur_depth == self.depth:
            return None, self.evaluationFunction( gameState )


        best_action, best_score = None, None

        if agent_index == 0:  # max player
            action_list = gameState.getLegalActions(agent_index)
            best_score = float('-inf')
            for action in action_list:
                next_state = gameState.getNextState( agent_index, action )
                _, score = self.alphaBeta( cur_depth, agent_index+1, next_state, alpha, beta )

                if score > best_score:
                    best_score = score
                    best_action = action

                alpha = max( alpha, score )
                if alpha > beta:
                    break

        else:  # min player
            action_list = gameState.getLegalActions(agent_index)
            best_score = float('inf')
            for action in action_list:
                next_state = gameState.getNextState( agent_index, action )
                _, score = self.alphaBeta( cur_depth, agent_index+1, next_state, alpha, beta )

                if score < best_score:
                    best_score = score
                    best_action = action

                beta = min(beta, score)
                if beta < alpha:
                    break
        
        return best_action, best_score
     # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        action, score = self.expectimax( 0, 0, gameState )
        return action

    def expectimax( self, cur_depth, agent_index, gameState ):
        """
        Same code with minimax, with slightly different.
        
        All call expectimax function as next layer,
        but the return value will have different traits according to the player:
            1. pacman: return max_action, max_score
            2. ghost : return min_action, average_score 
        """
        if gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction( gameState )

        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            cur_depth += 1

        if cur_depth == self.depth:
            return None, self.evaluationFunction( gameState )

        action_list = gameState.getLegalActions(agent_index)
        score_list = []
        for action in action_list:
            next_state = gameState.getNextState( agent_index, action ) 
            _, score = self.expectimax( cur_depth, agent_index+1, next_state )
            score_list.append( score )

        if agent_index == 0:
            return action_list[ score_list.index( max(score_list) ) ], max(score_list)
        else:
            return action_list[ score_list.index( min(score_list) ) ], sum( score_list ) / float(len(score_list))
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    """
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    * In order to achieve higher than 1000 points,              *
    * we need to earn extra 200 point by eating a scared ghost. *
    * Thus, we need to do ghost hunting when a ghost is scared. *
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

    1. game win :  inf
       game lose: -inf

    2. If the pacman position is on the
       ghost       : -inf to avoid die
       scared ghost:  inf ghost hunting

    3. The score is consist of
        a: The nearest food step - BFS
        b: manhattanDistance of active ghost
        c: manhattanDistance of scared ghost with scaredTimer > 2
        d: left food count
        f: current state score counted by the game

        score = (
                  -a * 0.05 +       #  nearest, better. Food is 10 points, scared ghost is 200 points, thus the weight of food is light.
                  +b * 0.7  +       #  further, better, 0.7 is a suitable weight 
                  100 - c   +       #  closer, better
                  -d * 30   +       #  left food penalty
                  e                 #  higher current score, better  (to avoid wasting time)
                              )
    """
    if currentGameState.isWin():
        return float('inf')
    elif currentGameState.isLose():
        return float('-inf')

    pos = currentGameState.getPacmanPosition()
    for ghost in currentGameState.getGhostStates():
        if pos == ghost.getPosition():
            if ghost.scaredTimer == 0:
                return float('-inf')
            elif ghost.scaredTimer > 0:
                return float('inf')

    score = 0
    q = util.Queue()
    q.push( (pos, 0) )
    capsulePos = currentGameState.getCapsules()
    currentFoodMap = currentGameState.getFood()
    vis = []
    while not q.isEmpty():
        curPos, step = q.pop()
        for dir in [ (-1, 0), (0, -1), (1, 0), (0, 1) ]:  # up, left, down, right
            newRow = curPos[0] + dir[0]
            newCol = curPos[1] + dir[1]
            newPos = (newRow, newCol)
            if newPos in vis:
                continue
            if currentFoodMap[ newRow ][ newCol ] == True or newPos in capsulePos:  # find the food
                score = -(step+1)*0.05  # a
                break
            elif currentGameState.hasWall(newRow, newCol) == False:
                q.push( (newPos, step+1) )
                vis.append( newPos )
    
    ghosts = currentGameState.getGhostStates()
    for ghost in ghosts:
        if ghost.scaredTimer == 0:
            score += manhattanDistance(ghost.getPosition(), pos)*0.7  # b
        else:
            if ghost.scaredTimer > 2:
                score += (100-manhattanDistance(ghost.getPosition(), pos))    # c

    score -= 30 * (currentGameState.getFood().count())    # d
    score += currentGameState.getScore()*1    # e
    return score
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
