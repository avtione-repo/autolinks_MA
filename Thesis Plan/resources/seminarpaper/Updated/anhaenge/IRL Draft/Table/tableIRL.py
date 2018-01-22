# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:43:35 2015

@author: cruz

"""
# Libraries Declaration
import numpy as np
import matplotlib.pyplot as plt
from classes.Variables import *
from classes.Scenario import Scenario
from classes.Agent import Agent
from classes.DataFiles import DataFiles

resultsFolder = 'results/'
files = DataFiles()


def setupRewardsPlot():
    dataRL = np.genfromtxt(resultsFolder + 'rewardsRL.csv', delimiter=',')
    dataIRL = np.genfromtxt(resultsFolder + 'rewardsIRL.csv', delimiter=',')
    meansRL = np.mean(dataRL, axis=0)
    meansIRL = np.mean(dataIRL, axis=0)

    convolveSet = 50
    convolveRL = np.convolve(meansRL, np.ones(convolveSet) / convolveSet)
    convolveIRL = np.convolve(meansIRL, np.ones(convolveSet) / convolveSet)

    plt.rcParams['font.size'] = 16
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.figure('Collected reward')
    plt.suptitle('Collected reward')

    plt.plot(meansIRL, label='Average reward IRL', linestyle='--', color='r')
    plt.plot(meansRL, label='Average reward RL', linestyle='--', color='y')

    plt.plot(convolveIRL, linestyle='-', color='0.2')
    plt.plot(convolveRL, linestyle='-', color='0.2')

    plt.legend(loc=4, prop={'size': 12})
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.grid()

    my_axis = plt.gca()
    # my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
    my_axis.set_xlim(convolveSet, len(meansRL))




# end of plotRewards method

def setupStepsPlot():
    dataRL = np.genfromtxt(resultsFolder + 'stepsRL.csv', delimiter=',')
    dataIRL = np.genfromtxt(resultsFolder + 'stepsIRL.csv', delimiter=',')
    meansRL = np.mean(dataRL, axis=0)
    meansIRL = np.mean(dataIRL, axis=0)

    convolveSet = 50
    convolveRL = np.convolve(meansRL, np.ones(convolveSet) / convolveSet)
    convolveIRL = np.convolve(meansIRL, np.ones(convolveSet) / convolveSet)

    plt.rcParams['font.size'] = 16
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    plt.figure('Steps taken')
    plt.suptitle('Steps taken')

    plt.plot(meansIRL, label='Average steps IRL', linestyle='--', color='r')
    plt.plot(meansRL, label='Average steps RL', linestyle='--', color='y')

    plt.plot(convolveIRL, linestyle='-', color='0.2')
    plt.plot(convolveRL, linestyle='-', color='0.2')

    plt.legend(loc=4, prop={'size': 12})
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.grid()

    my_axis = plt.gca()
    # my_axis.set_ylim(Variables.punishment-0.8, Variables.reward)
    my_axis.set_xlim(convolveSet, len(meansRL))




# end of plotSteps method


# Feedback parameter meaning differs depending on feedback strategy.
# For random advising: advice probability in every state
# For early advising, importance advising and mistake correcting: Advice budget per episode
def trainAgent(tries, episodes, scenario, teacherAgent=None, feedbackStrategy=0, feedbackParameter=0.0):
    if teacherAgent == None:
        filenameSteps = resultsFolder + 'stepsRL.csv'
        filenameRewards = resultsFolder + 'rewardsRL.csv'
    else:
        filenameSteps = resultsFolder + 'stepsIRL.csv'
        filenameRewards = resultsFolder + 'rewardsIRL.csv'

    files.createFile(filenameSteps)
    files.createFile(filenameRewards)

    for i in range(tries):
        print('Training agent number: ' + str(i + 1))
        agent = Agent(scenario)
        [steps, rewards] = agent.train(episodes, teacherAgent, feedbackStrategy, feedbackParameter)

        files.addToFile(filenameSteps, steps)
        files.addFloatToFile(filenameRewards, rewards)
    # endfor

    return agent


# end trainAgent method

if __name__ == "__main__":
    print("Interactive RL for cleaning a table is running ... ")
    tries = 30
    episodes = 1000

    scenario = Scenario()

    # Training with autonomous RL
    print('RL is now training the teacher agent with autonomous RL')
    teacherAgent = trainAgent(tries, episodes, scenario)

    # Training with interactive RL

    # feedbackParameter = 0.3
    # feedbackStrategy = feedbackstrategy_random;
    #
    # feedbackParameter = 3
    # feedbackStrategy = feedbackstrategy_early;
    #
    # feedbackParameter = 4
    # feedbackStrategy = feedbackstrategy_importance;
    #
    feedbackParameter = 3
    feedbackStrategy = feedbackstrategy_correction;

    print('IRL is now training the learner agent with interactive RL')
    learnerAgent = trainAgent(tries, episodes, scenario, teacherAgent, feedbackStrategy, feedbackParameter)
    setupStepsPlot()
    setupRewardsPlot()
    plt.show()

print("The end")  # end of main method
