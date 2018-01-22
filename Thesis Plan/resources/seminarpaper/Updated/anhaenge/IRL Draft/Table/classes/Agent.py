# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 17:43:35 2015

@author: cruz
"""

import numpy as np
import Variables


class Agent(object):
    alpha = 0.3  # 0.1 #0.7
    gamma = 0.9  # 0.4
    epsilon = 0.1  # 0.25

    def __init__(self, scenario):
        self.scenario = scenario
        self.numberOfStates = self.scenario.getNumberOfStates()
        self.numberOfActions = self.scenario.getNumberOfActions()
        self.Q = np.random.uniform(0.0, 0.01, (self.numberOfStates, self.numberOfActions))
        self.feedbackAmountTotal = 0
        self.feedbackAmountEpisode = 0

    # end of __init__ method

    def selectAction(self, state):
        if (np.random.rand() <= self.epsilon):
            action = np.random.randint(self.numberOfActions)
        else:
            action = np.argmax(self.Q[state, :])
        # endIf
        return action

    # end of selectAction method

    def actionByFeedback(self, state, teacherAgent, feedbackStrategy, feedbackParameter):
        if (feedbackStrategy == Variables.feedbackstrategy_random):
            if (np.random.rand() < feedbackParameter):
                # get advice
                action = np.argmax(teacherAgent.Q[state, :])
                self.feedbackAmountEpisode += 1
            else:
                action = self.selectAction(state)
        elif (feedbackStrategy == Variables.feedbackstrategy_early):
            if (self.feedbackAmountEpisode < feedbackParameter):
                # get advice
                action = np.argmax(teacherAgent.Q[state, :])
                self.feedbackAmountEpisode += 1
            else:
                action = self.selectAction(state)

        elif (feedbackStrategy == Variables.feedbackstrategy_importance):
            if (self.feedbackAmountEpisode < feedbackParameter and teacherAgent.isImportant(state)):
                # get advice
                action = np.argmax(teacherAgent.Q[state, :])
                self.feedbackAmountEpisode += 1
            else:
                action = self.selectAction(state)

        elif (feedbackStrategy == Variables.feedbackstrategy_correction):
            teacheraction = np.argmax(teacherAgent.Q[state, :])
            selfaction = self.selectAction(state)
            if (self.feedbackAmountEpisode < feedbackParameter and teacherAgent.isImportant(state) and teacheraction != selfaction):
                # get advice

                action = teacheraction

                self.feedbackAmountEpisode += 1
            else:
                action = selfaction

        ####
        # endIf
        return action

    # end of actionByFeedback

    def isImportant(self, state):

        # importance by range
        # importance = max(self.Q[state, :]) - min(self.Q[state, :])

        # importance by total deviation
        importance = np.mean(np.absolute(self.Q[state, :] - np.mean(self.Q[state,:])))
        # print(importance)



        if importance > Variables.importancethreshold:
            return 1

        else:
            return 0

    # end of isImportant

    def train(self, episodes, teacherAgent=None, feedbackStrategy=0, feedbackParameter=0):
        contCatastrophic = 0
        contFinalReached = 0
        steps = np.zeros(episodes)
        rewards = np.zeros(episodes)

        for i in range(episodes):
            contSteps = 0
            accReward = 0
            self.feedbackAmountEpisode = 0
            self.scenario.resetScenario()
            state = self.scenario.getState()
            action = self.actionByFeedback(state, teacherAgent, feedbackStrategy, feedbackParameter)

            # expisode
            while True:
                # perform action
                self.scenario.executeAction(action)
                contSteps += 1

                # get reward
                reward = self.scenario.getReward()
                accReward += reward
                # catastrophic state

                stateNew = self.scenario.getState()

                if reward == Variables.punishment:
                    contCatastrophic += 1
                    self.Q[state, action] = -0.1
                    break

                actionNew = self.actionByFeedback(stateNew, teacherAgent, feedbackStrategy, feedbackParameter)

                # updating Q-values
                self.Q[state, action] += self.alpha * (reward + self.gamma *
                                                       self.Q[stateNew, actionNew] -
                                                       self.Q[state, action])

                if reward == Variables.reward:
                    contFinalReached += 1
                    break

                state = stateNew
                action = actionNew
            # end of while
            steps[i] = contSteps
            rewards[i] = accReward
            self.feedbackAmountTotal += self.feedbackAmountEpisode
        # end of for
        print("Total feedback amount in all episodes:" + str(self.feedbackAmountTotal))
        print("Total amount of failures" + str(contCatastrophic))
        print("Total amount of successes" + str(contFinalReached))

        return steps, rewards
        # end of train method

# end of class Agent
