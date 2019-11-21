import numpy as np
import random

# This class is for a gridworld object
# It stores all information and method related to the actual grid
class gridworld():
    def __init__(self):
        self.i = 3  # initial position assuming 0, 0 is top left corner
        self.j = 0  # initial position assuming 0, 0 is top left corner
        self.states = []

        # A list of all the possible states
        for i in range(7):
            for j in range(10):
                self.states.append((i, j))

    # A move on the grid, either right, up-right, up, up-left, left, down-left, down, down-right
    # Takes wind into account
    def moveState(self, action):
        if action == 0: # move right
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
            if self.j < 9:
                self.j += 1
        elif action == 1: # move up-right
            if self.i > 0 and self.j < 9: # check out of bounds
                self.i -= 1
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
            if self.i > 0 and self.j < 9: # check out of bounds
                self.j += 1
        elif action == 2: # move up
            if self.i > 0: # check out of bounds
                self.i -= 1
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
        elif action == 3: # move up-left
            if self.i > 0 and self.j > 0: # check out of bounds
                self.i -= 1
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
            if self.i > 0 and self.j > 0:
                self.j -= 1
        elif action == 4: # move left
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
            if self.j > 0:
                self.j -= 1
        elif action == 5: # move down-left
            if self.i < 6 and self.j > 0: # check out of bounds
                self.i += 1
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
            if self.i < 6 and self.j > 0:
                self.j -= 1
        elif action == 6: # move down
            if self.i < 6: # check out of bounds
                self.i += 1
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
        elif action == 7: # move down-right
            if self.i < 6 and self.j < 9: # check out of bounds
                self.i += 1
            if self.i > 0 and self.j in (3, 4, 5, 8, 6, 7): # check out of bounds
                wind = random.choice([1, 0, -1]) 
                if self.j in (3, 4, 5, 8): # wind factor of 1
                    wind -= 1
                elif self.j == 6 or self.j == 7: # wind factor of 2
                    wind -= 2
                if self.i + wind < 0:
                    self.i = 0
                else:
                    self.i += wind
            if self.i < 6 and self.j < 9:
                self.j += 1

    # Get the reward for the specific state
    def getReward(self, nextState):
        # If we have reached the goal state, the game ends and we get a reward of 1
        # Otherwise, return the reward of -1
        if nextState == (3, 7):
            return 1
        return -1

# This class lets us create a SARSA game object
class SARSA():
    def __init__(self, gamma, alpha, epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.qTable = np.zeros((70, 8)) # initialize the q-table as zeros

    def algorithm(self):
        # Loop through the number of episodes
        converged = False # flag variable to see if algorithm has converged
        while converged == False:

            converged = True
            stateList = [] # list of states to be printed at the end as the trajectory
            grid = gridworld() # initialize grid
            s = (grid.i, grid.j) # get current state
            states = grid.states
            index = states.index(s) # find the index of the current state in the list of states, so we can find its location in the q-table

            actionValues = self.qTable[index] # find state in q - table

            randNum = random.random() # random number generated for e-greedy action selection

            if randNum > self.epsilon: # take the greedy action
                egreedyAction = (np.ndarray.tolist(actionValues)).index(max(actionValues))
            else: # take a random action
                egreedyAction = random.randint(0, 7)

            # Iterate through a single game until the goal is reached
            while (grid.i, grid.j) != (3, 7):
                stateList.append(s) # add current state to list of states
                stateList.append(egreedyAction)  # add current action to list of states

                index = states.index(s) # find index of current state
                grid.moveState(egreedyAction) # move to the next state
                s2 = (grid.i, grid.j) # get next state (s')
                reward = grid.getReward(s2) # get reward of next state (s')

                index2 = states.index(s2) # get index of next state (s')

                actionValues = self.qTable[index2] # find Q(s', a') for all a'

                randNum = random.random() # random number of e-greedy action selection

                if randNum > self.epsilon: # select the greedy action
                    egreedyAction2 = (np.ndarray.tolist(actionValues)).index(max(actionValues))
                else: # select a random action
                    egreedyAction2 = random.randint(0, 7)

                # update Q(S,A) using SARSA
                newQ = self.alpha * (reward + self.gamma * self.qTable[index2][egreedyAction2] - self.qTable[index][egreedyAction])
                self.qTable[index][egreedyAction] = self.qTable[index][egreedyAction] + newQ

                # if the change in Q-value is greater than a given threshold, then the game has not converged
                if newQ > 0.0001 or newQ < -0.0001:
                    converged = False

                s = s2 # s = s'
                egreedyAction = egreedyAction2 # a = a'

        print(stateList)

    def optimalPolicy(self):
        optimalPolicy = np.zeros((7, 10))
        for i in range(70):
            row = i // 7
            column = i % 7
            # obtain the optimal policy for a state from its corresponding value in the q-table
            optimalPolicy[column][row] = (np.ndarray.tolist(self.qTable[i])).index(max(self.qTable[i]))
        print(optimalPolicy)

def main():
    agent = SARSA(0.5, 0.5, 0.1)
    agent.algorithm()
    agent.optimalPolicy()

main()
