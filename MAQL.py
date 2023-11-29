
import numpy as np
# from pyibl import Agent
import random as random
import itertools
from math import exp
import math 
import sys 
from collections import deque
from itertools import count
# from speedyibl import Agent 

class AgentQL(object):
    mkid = next(itertools.count())

    def __init__(self, config):
        
        self.default_utility =  0.1
        self.c = config
# 		self.select_position()
        self.outcomes = {}
        # self.alpha = 0.1
        self.epsilon = self.c.eps.initial
        self.__ep = 0
        self.goods = 0
        '''
        :param int agentID: Agent's ID
        :param dict config: Dictionary containing hyperparameters
        '''
        self.episodeCounter = 0
        self.c.id = AgentQL.mkid
        self.Temps = {}

    def generate_outcomes(self, s_hash):
        self.outcomes[s_hash] = [self.default_utility/2]*self.c.outputs
    
    def move(self, o, explore=True):
        # '''
        # Returns an action from the ibl agent instance.
        # '''
        
        if self.episodeCounter > self.__ep:
            self.epsilon = max((self.epsilon * self.c.eps.discount), 0)
        if self.episodeCounter > self.__ep:
            self.__ep += 1
        
        s_hash = o.tobytes()
        if (s_hash) not in self.outcomes:
            self.last_action = random.randrange(self.c.outputs)
            self.generate_outcomes(s_hash)
        elif explore and random.random() < self.epsilon:
            self.last_action = random.randrange(self.c.outputs)
        else:
            self.last_action = self.choose_td(s_hash)
        
        self.option = (s_hash, self.last_action)

        self.s = s_hash
        

        return self.last_action



    def feedback(self, reward, terminal, o):
        # '''
        # Feedback is passed to the deep rl agent instance.
        # :param float: Reward received during transition
        # :param boolean: Indicates if the transition is terminal
        # :param tensor: State/Observation
        # '''
        
        s_hash = o.tobytes()
        if self.c.mamethod == 'leniency':
            if self.s not in self.Temps:
                self.Temps[self.s] = np.ones(self.c.outputs)*self.c.len.max
            temp_action = self.Temps[self.s][self.last_action]
            self.leniency = 1 - np.exp(-self.c.len.theta*temp_action)

        if terminal:
            self.episodeCounter += 1
# 		
        # self.respond(None)
    
        if s_hash == self.s:
            outcome = reward
        else:
            if terminal > 0:
                best_utility = 0
            elif (s_hash) in self.outcomes:
                utilities = self.outcomes[s_hash] #self.blend_compute(self.t, s_hash)
                best_utility = max(utilities) #max(utilities,key=lambda x:x[0])[0]
            else:
                best_utility = self.default_utility

            outcome = reward + self.c.gamma*best_utility - self.outcomes[self.s][self.last_action]
        
        if self.c.mamethod == 'leniency':
            if outcome > 0 or random.random() > self.leniency: # self.drl.replay_memory._episode[-1][7]:
                self.outcomes[self.s][self.last_action] += self.c.alpha*outcome

            if (s_hash) not in self.Temps:
                temp_mean = self.c.len.max
            else:
                temp_mean = np.mean(self.Temps[s_hash])
            if terminal:
                self.Temps[self.s][self.last_action] = self.c.len.delta*self.Temps[self.s][self.last_action]
            else:
                
                self.Temps[self.s][self.last_action] = self.c.len.delta*((1-self.c.len.tau)*self.Temps[self.s][self.last_action] + self.c.len.tau*temp_mean)
            
        elif self.c.mamethod == 'hysteretic':
            if outcome > 0:
                self.outcomes[self.s][self.last_action] += self.c.alpha*outcome
            else:
                self.outcomes[self.s][self.last_action] += self.c.hys.beta*outcome
        else:
            self.outcomes[self.s][self.last_action] += self.c.alpha*outcome
    
    def blend_compute(self,s_hash):
        outcomes = self.outcomes[s_hash]
        blends = []
        for a,i in zip(range(self.c.outputs),count()):
            blends.append((outcomes[a],i))
            
        return blends 

    def choose_td(self, s_hash):
        utilities = self.blend_compute(s_hash)
        best_utility = max(utilities,key=lambda x:x[0])[0]
        best = random.choice(list(filter(lambda x: x[0]==best_utility,utilities)))[1]
        return best 
    
    
    def opt(self):
        '''
        Called to optimise the agent.
        '''
        pass 

