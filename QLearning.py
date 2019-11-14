'''
Written by Ella Stewart on 2019-11-12
Implements QLearning for the Windy Grid World
'''
import random

'''
Sets up the grid to be used for Qlearning
Initializes coordinates as start state
Moves states given an action
'''
class gridWorld:

    # Constructor that initializes start state and all possible states
    def __init__(self):
        self.i = 0
        self.j = 3
        self.states = []
        for x in range(10):
            for y in range(7):
                self.states.append((x, y))

    # Moves i and j given an action
    def moveState(self, action):
        if action == 0:  # Move right
            if self.i < 9:
                self.i += 1
        elif action == 1:  # Move up
            if self.j < 6:
                self.j += 1
        elif action == 2:  # Move left
            if self.i > 0:
                self.i -= 1
        else:
            if self.j > 0:  # Move down
                self.j -= 1

    # Returns current state as tuple
    def getCurrentState(self):
        currentState = (self.i, self.j)
        return currentState

    # Accessor for the states
    def getStates(self):
        return self.states

    # Defines rewards for each state
    def getReward(self, state):
        if state == (7, 3):
            reward = 1
        else:
            reward = -1
        return reward

'''
Used to implement the QLearning algorithm
Creates instance of grid
Initializes qTable
'''
class QLearning:

    # Adjustable gamma, alpha and epsilon parameters
    def __init__(self, gamma, alpha, epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # 70 x 4 qTable
        # 70 states, 4 actions
        self.qTable = []
        for i in range(70):
            self.qTable.append([0, 0, 0, 0])

    # Called to the the QLearning algorithm
    def runAlgorithm(self):

        # Large number of episodes
        for i in range(10000):
            print(i)
            # Print states visited in order in final round
            if i == 9999:
                finalStates = []
            # Initializing grid and getting current state
            grid = gridWorld()
            S = grid.getCurrentState()
            if i == 9999:
                finalStates.append(S)
            states = grid.getStates()

            # Episode ends at goal state (3,7) on grid
            while grid.getCurrentState() != (7, 3):
                # Get the possible actions
                index1 = states.index(S)
                actionValues = self.qTable[index1]

                # Determines greedy or random
                num = random.uniform(0, 1)

                if num <= self.epsilon:
                    # Random action
                    randomChoices = [0, 1, 2, 3]
                    action = random.choice(randomChoices)
                else:
                    # Greedy action
                    action = actionValues.index(max(actionValues))

                # Observe S' and R
                grid.moveState(action)
                S2 = grid.getCurrentState()
                reward = grid.getReward(S2)

                # Append state to list if final episode
                if i == 9999:
                    finalStates.append(S2)

                # Get possible A'
                index2 = states.index(S2)
                possibleActions = self.qTable[index2]

                maxA = possibleActions.index(max(possibleActions))
                qSA = self.qTable[index1][action]
                # Update QTable
                self.qTable[index1][action] = qSA + self.alpha*(reward + self.gamma*maxA - qSA)
                S = S2  # S <- S'

        return finalStates

def main():

    agent = QLearning(0.5, 0.5, 0.3)
    finalStates = agent.runAlgorithm()
    print(finalStates)

main()















