# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0 # start with V(s) = 0 for all states

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() #return a list of states

        for k in range(self.iterations):
          # make a copy of the values in the previous iteration
          old_values = self.values.copy()
          for state in states:
            if self.mdp.isTerminal(state):
              pass
            else:
              # all possible actions of the current states
              actions = self.mdp.getPossibleActions(state)
              # initialize the max Q value (i.e. value of current state)
              Q_list = []
              for action in actions:
                nextState_prob_list = self.mdp.getTransitionStatesAndProbs(state, action)
                nextStates, transProbs = zip(*nextState_prob_list)
                Q = 0
                for nextState, transProb in nextState_prob_list:
                  reward = self.mdp.getReward(state, action, nextState)
                  Q += transProb * (reward + self.discount * old_values[nextState])
                Q_list.append(Q)
              self.values[state] = max(Q_list)



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        nextState_prob_list = self.mdp.getTransitionStatesAndProbs(state, action)
        nextStates, transProbs = zip(*nextState_prob_list)
        Q = 0
        for nextState, transProb in nextState_prob_list:
            reward = self.mdp.getReward(state, action, nextState)
            Q += transProb * (reward + self.discount * self.values[nextState])
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # check if the current state is the terminal state
        if self.mdp.isTerminal(state):
          return None
        else:
          # policy = ''
          # Qmax = 0
          action_list = []
          Q_list = []
          for action in self.mdp.getPossibleActions(state):
            action_list.append(action)
            Q = self.computeQValueFromValues(state, action)
            Q_list.append(Q)
          Qmax_ind = Q_list.index(max(Q_list)) # index of max Q value in the list
          policy = action_list[Qmax_ind]
          return policy
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
