import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from itertools import combinations
import random

from Constants import Constants

#random.seed()  #choose your lucky number here and fix seed

#from QuatumEnvironment_dummy import QuantumEvnironment     #import the distributed quantum computing simulation environment
from QuatumEnvironment import QuantumEvnironment     #import the distributed quantum computing simulation environment

import copy
import csv


from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
        

class EnvUpdater(gym.Env):      #gym is an opanAI's environment generator tools. however, we doing most of it ourselves for the distributed qunatum computing game
    environment_name = "distQuantComp"
    """


    """

    def __init__(self, completion_deadline): 

        self.quantumEnv = QuantumEvnironment()
        self.state = self.quantumEnv.state   #state at the beginning decided on by the processor and DAG configurations
        self.mask = self.quantumEnv.mask   #mask at the beginning decided on by the processor and DAG configurations, value always 1 if no masking used
        
        self.action_dim = self.quantumEnv.generate_action_size()  #environment provides action space size
        self.action_list = spaces.Discrete(self.action_dim)   #discrete action space
        self.action_space = self.action_list
        
        # print("state is: ",  self.state)
        # print("action space  is: ", self.action_list)
        
        self.trials = 1
        self.numSteps = completion_deadline
        self.stepCount = 0
        self.dummy_stepCount = 0
        self.EpiCount = 0
        self.successfulGames = 0
        
        self.epiTotalREward = 0
        self.reward_filename = 'rewards.csv'
        with open(self.reward_filename, 'a', newline="") as file:
            writer = csv.writer(file)
            #writer.writerow(["Episode","Total"])
        
        self.done_filename = 'doneTime.csv'    
        with open(self.reward_filename, 'a', newline="") as file2:
            writer2 = csv.writer(file2)
            #writer.writerow(["Episode","Total"])
        
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        

    def step(self, action): #this is the main function where each step of the game takes place, i.e. synchroniztion step
        
        
    
        reward, new_state, new_mask, successfulDone = self.quantumEnv.RL_step(action)
        self.state = new_state 
        self.mask = new_mask
        steptemp = copy.deepcopy(self.stepCount)
        steptemp_dummy = copy.deepcopy(self.dummy_stepCount)
        
        
        done = self.deadline_monitor(successfulDone) #deadline monitoring
        #print(self.stepCount)
        
        # if reward >= Constants.REWARD_SCORE:
        #     print("a solution at step: ", steptemp)
        #     print("Dummy Step Count: ", steptemp_dummy)
            
        if done and not successfulDone:
            # print("Dummy Step Count: ", steptemp_dummy)
            reward += Constants.REWARD_DEADLINE
            
        self.epiTotalREward += reward
        if successfulDone:
            self.successfulGames += 1
            #print("total_success: ",self.successfulGames)
        
        if done:
            #print("epiCount: ",self.EpiCount)
            self.EpiCount += 1
            #print("solved/done after step number: ", steptemp)
            row = [self.EpiCount, self.epiTotalREward]
            row2 = [self.EpiCount, steptemp]
            append_list_as_row(self.reward_filename, row)
            append_list_as_row(self.done_filename, row2)
            self.epiTotalREward = 0
        
        self.dummy_stepCount += 1
        if action == 0: ### action=0 is always a stop, and that is the only increase in step
            self.stepCount += 1
        
        # print("action: ", action)
        # print("new_state: ", new_state)
        # print("new_mask: ", new_mask)
        # print("reward: ", reward)
 
        return new_state, new_mask, reward, done, {}       #supposed to return new state, reward and done to the learning agent


    def reset(self):
        self.quantumEnv.environment_reset()  
        self.state = self.quantumEnv.state
        self.mask = self.quantumEnv.mask
        #print("reset_was_called")
        return np.array(self.state), np.array(self.mask)


    def deadline_monitor(self, successfulDone):
        if self.numSteps < self.stepCount or successfulDone:   
            done = True
            self.stepCount = 0
            self.dummy_stepCount = 0
        else:
            done = False
        return done
    



