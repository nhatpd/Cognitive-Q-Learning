import numpy as np
import random
import cv2
import copy

class CTF(object):
    """ Team capture the flag. """
    def __init__(self, version, number_agents, deliver = True):
        '''
        :param version: Integer specifying which configuration to use
        '''
        if version == 1: # Standard
            from .envconfig_v1 import EnvConfigV1
            self.c = EnvConfigV1(number_agents)

        # Fieldnames for stats
        self.fieldnames = ['Episode',
                           'Steps',
                           'blue_team',
                           'red_team',
                           'Coop_Transport_blue_Steps',
                            'Coop_Transport_red_Steps',
                            'Ind_blue_A1_Steps',
                            'Ind_blue_A2_Steps',
                            'Ind_red_A1_Steps',
                            'Ind_red_A2_Steps',
                           'Defended',
                           'Time']

        self.__dim = self.c.DIM     # Observation dimension
        self.__out = self.c.ACTIONS # Number of actions
        self.episode_count = 0      # Episode counter
        self.time = 0
        # Used to add noise to each cell
        self.ones = np.ones(self.c.DIM, dtype=np.float64)
        DIM = np.copy(self.c.DIM)
        self.DIM = np.append(DIM,3)
        # self.im = np.ones(DIM,dtype = np.float64)*255
        self.im = np.zeros(self.DIM,dtype = np.float64)
        self.colors_agents = [(255.0, 0.0,   0.0), (0.0,   0.0,   255.0)]
        self.colors_foods = [(0.0,   0.0,   100.0), (100.0, 0.0,   0.0)]
        
        self.deliver = deliver
    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    def render(self):
        '''
        Used to render the env.
        '''
        r = 16 # Number of times the pixel is to be repeated
        try:
            # im = np.zeros(self.DIM, dtype=np.float64)
            # for y, x in self.c.OBSTACLES_YX:
            #     im[y][x] = (145, 145, 145)
            # colors = [(255.0, 0.0,   0.0), (0.0,   0.0,   255.0)]
            im = copy.deepcopy(self.im)
            for i in range(2):
                for j in range(self.c.NUMBER_OF_AGENTS_per_TEAM):
                    # if (self.agents_xy[i][j][0] < self.c.MID and i == 0) or\
                    #     (self.agents_xy[i][j][0] > self.c.MID and i == 1):
                    im[self.agents_yx[i][j][0]][self.agents_yx[i][j][1]] = self.colors_agents[i]
            # colors_foods = [(0.0,   0.0,   100.0), (100.0, 0.0,   0.0)]
            for i in range(2):
                im[self.foods_yx[i][0]][self.foods_yx[i][1]] = self.colors_foods[i]
            
            # im = self.upObservations(0,True)[0]
            img = np.repeat(np.repeat(im, r, axis=0), r, axis=1).astype(np.uint8)
            # img = np.repeat(np.repeat(self.s[0], r, axis=0), r, axis=1).astype(np.uint8)
            # cv2.imshow('image', img)
            cv2.imshow('CTF', img)
            k = cv2.waitKey(1)
            if k == 27:         # If escape was pressed exit
                cv2.destroyAllWindows()

            
        except AttributeError:
            pass

    def stats(self):
        '''
        Returns stats dict
        '''
        stats = {'Episode': str(self.episode_count), 
                 'Steps': str(self.steps), 
                 'blue_team': str(self.rewards[0]), # blue team index: 0
                 'red_team': str(self.rewards[1]), # red team index: 1
                 'Coop_Transport_blue_Steps':str(self.coopTransportSteps[0]),
                 'Coop_Transport_red_Steps': str(self.coopTransportSteps[1]),
                 'Ind_blue_A1_Steps':str(self.individualSteps[0][0]),
                 'Ind_blue_A2_Steps':str(self.individualSteps[0][1]),
                 'Ind_red_A1_Steps':str(self.individualSteps[1][0]),
                 'Ind_red_A2_Steps':str(self.individualSteps[1][1]),
                 'Defended':str(self.defended),
                 'Time':str(self.time)}
        return stats

    def reset(self, color_images = [False, False]):
        '''
        Reset everything. 
        '''
        
        self.s_t = np.zeros(self.c.DIM, dtype=np.float64)
         
        
        # Obstacles, agents and goods are initialised:
        self.setObstacles()
        self.initFlags()
        self.initAgents()
        for y, x in self.c.OBSTACLES_YX:
                self.im[y][x] = (145, 145, 145)
        # self.setCapsules()
        # self.setSZones()
        # self.initFlags()
        self.delivered = False
        # Used to keep track of the reward total acheived throughout 
        # the episode:
        self.reward_total = 0.0

        # Episode counter is incremented:
        self.episode_count += 1

        #initial rewards
        self.rewards = [0, 0]
        

        #Initialize flag holding
        self.holding_flags = {}
        #Initialize capsules eaten
        self.eaten_capsules = {} # [False, False, False, False]
        self.eaten_capsules_moves = [0, 0]
        for j in range(2):
            self.holding_flags[j] = [False for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM)]
            self.eaten_capsules[j] = [False for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM)]
           
        self.isPacman = [False, False, False, False]
        # For statistical purposes:
        # Step counter for the episode is initialised
        self.steps = 0 
        self.t = False 

        # Moves taken in the same direction while carrying the goods
        self.coopTransportSteps = [0, 0]

        # # Moves taken in the same direction while carrying the goods
        # self.coordinatedTransportSteps = [0,0]

        self.individualSteps = [[0 for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM)], [0 for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM)]]
        
        
        return [self.upObservations(0, color_images[0]), self.upObservations(1, color_images[1])]

    def terminal(self):
        '''
        Find out if terminal conditions have been reached.
        '''
        # if self.rewards[0] >= len(self.c.FOODS_YX)/2 or self.rewards[1] >= len(self.c.FOODS_YX)/2:
        #     self.t = True 
        return self.t

    def step(self, actions, orders, color_images = [False, False]):
        '''
        Change environment state based on actions.
        :param actions: list of integers
        '''
        # Agents move according to actions selected
        self.r = [[0 for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM)],[0 for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM)]]
        
        for i in range(2):
            self.moveAgents(actions[i], orders[i])
            # self.moveAgents(actions, orders)

        
        self.steps += 1 
        
        
        return [self.upObservations(0,color_images[0]),self.upObservations(1,color_images[1])], self.r, self.terminal() 
 # self.terminal() 

    def upObservations(self, team_id, color_image):
        
        if color_image:
            
            s = [copy.deepcopy(self.im) for ii in range(self.c.NUMBER_OF_AGENTS_per_TEAM)]
        else: 
            s = [copy.deepcopy(self.s_t) for ii in range(self.c.NUMBER_OF_AGENTS_per_TEAM)]
        for j in range(self.c.NUMBER_OF_AGENTS_per_TEAM):
            if color_image:
                for i in range(2):
                    s[j][self.foods_yx[i][0]][self.foods_yx[i][1]] = self.colors_foods[i]
            for jj in range(self.c.NUMBER_OF_AGENTS_per_TEAM):
                if color_image:
                    s[j][self.agents_yx[team_id][jj][0]][self.agents_yx[team_id][jj][1]] = self.colors_agents[team_id]
                else:
                    s[j][self.agents_yx[team_id][jj][0]][self.agents_yx[team_id][jj][1]] = self.c.AGENTS[team_id]
                # else:
                #     self.s[i][j][self.agents_xy[i][jj][0]][self.agents_xy[i][jj][1]] = self.c.AGENTS_Other_Side[i]
                if abs(self.agents_yx[team_id][j][0] - self.agents_yx[1-team_id][jj][0]) +\
                        abs(self.agents_yx[team_id][j][1] - self.agents_yx[1-team_id][jj][1]) <= 5:
                    # if (self.agents_yx[1-i][jj][0] < self.c.MID and i == 1) or\
                    #     (self.agents_xy[1-i][jj][0] > self.c.MID and i == 0):
                    if color_image:
                        s[j][self.agents_yx[1-team_id][jj][0]][self.agents_yx[1-team_id][jj][1]] = self.colors_agents[1-team_id]
                    else:

                        s[j][self.agents_yx[1-team_id][jj][0]][self.agents_yx[1-team_id][jj][1]] = self.c.AGENTS[1-team_id]
        return s
                            

    def initFlags(self):
        '''
        Goods position and carrier ids are initialised
        '''
      
        
        
        # Each goods has a delivered status that is initially set to false. 
        self.defended = True
        # self.t = False

         
        for y, x in self.c.FOODS_YX:
            self.s_t[y][x] = self.c.FOOD
        self.foods_yx = copy.deepcopy(self.c.FOODS_YX)
            

    def initAgents(self):
        '''
        Method for initialising the required number of agents and 
        positionsing them on designated positions within the grid
        '''

       
        self.agents_yx = copy.deepcopy(self.c.AGENTS_YX) # [copy.deepcopy(self.c.AGENTS_XY_B), copy.deepcopy(self.c.AGENTS_XY_R)]
        
    def setObstacles(self):
        '''
        Method used to initiate the obstacles within the environment 
        '''
        for y, x in self.c.OBSTACLES_YX:
            self.s_t[y][x] = self.c.OBSTACLE
            

    def flagsPickup(self, x, y, id):
        '''
        Method for picking up the tools, if the agents
        find themselves in positions adjecent to the goods.
        '''
        # print(x, self.c.MID,self.c.GW,self.c.GH)
        t = id//self.c.NUMBER_OF_AGENTS_per_TEAM
        aid = id % self.c.NUMBER_OF_AGENTS_per_TEAM

        if not self.holding_flags[t][aid]:
            #position above the food:
            if y == self.foods_yx[1-t][0]-1 and x == self.foods_yx[1-t][1]:
                self.holding_flags[t][aid] = True 
                # self.s_t[x][y] -= self.c.FOOD
                # print('test')
             #position below the food:
            elif y == self.foods_yx[1-t][0]+1 and x == self.foods_yx[1-t][1]:
                self.holding_flags[t][aid] = True 
            # print('test')
        

    def flagsDelivered(self, id_team):
        '''
        Method to check one of the goods 
        has been deliverd to the dropzone
        '''
        if sum(self.holding_flags[id_team]) >= 2:
            if (id_team == 0 and self.foods_yx[1-id_team][1] <= self.c.MID) or (id_team == 1 and self.foods_yx[1-id_team][1] >= self.c.MID-1):
                    self.rewards[id_team] += 1
                    r = 1
                    self.t = True 
            else:
                r = 0
        else:
            r = 0
        for i in range(self.c.NUMBER_OF_AGENTS_per_TEAM):
            self.r[id_team][i] += r

         

    def getNoisyState(self):
        ''' 
        Method returns noisy state.
        '''
        return self.s_t
        # return self.s_t + (self.c.NOISE * self.ones *\
                        #    np.random.normal(self.c.MU,self.c.SIGMA, self.c.DIM))

    def getObservations(self):
        '''
        Returns centered observation for each agent
        '''
        # observations = []
        # for i in range(self.c.NUMBER_OF_AGENTS):
        #     # Store observation
        #     observations.append(np.copy(self.getNoisyState()))
        # return observations 
        return self.s

    def getDelta(self, action):
        '''
        Method that deterimines the direction 
        that the agent should take
        based upon the action selected. The
        actions are:
        'Up':0, 
        'Right':1, 
        'Down':2, 
        'Left':3, 
        'NOOP':4
        :param action: int
        '''
        if action == 0:
            return 0, -1
        elif action == 1:
            return 1, 0    
        elif action == 2:
            return 0, 1    
        elif action == 3:
            return -1, 0 
        elif action == 4:
            return 0, 0   

    def moveAgents(self, actions, orders):
    #    '''
    #    Move agents according to actions.
    #    :param actions: List of integers providing actions for each agent
    #    '''
        
        action_a_holding = []
        targety_holding = []
        targetx_holding = []
        for action, order in zip(actions,orders):

            dx, dy = self.getDelta(action)
            t = order//self.c.NUMBER_OF_AGENTS_per_TEAM
            aid = order % self.c.NUMBER_OF_AGENTS_per_TEAM
            targetx = self.agents_yx[t][aid][1] + dx
            targety = self.agents_yx[t][aid][0] + dy
            
            if action == 4:
                self.r[t][aid] -= 0.01

            if self.holding_flags[t][aid]:
                self.PenaltyCollision(targetx, targety, t, aid)
                
                action_a_holding.append(action)
                targety_holding.append(targety)
                targetx_holding.append(targetx)

            else:
                self.individualSteps[t][aid] += 1
                if self.noCollision(targetx, targety, t, aid) and self.NoAgentCollision(targetx, targety, order):
                
                    self.moveAgent(order, targetx, targety)
                    self.flagsPickup(targetx, targety, order)
            
        if len(action_a_holding) >= 2 and action_a_holding[0] == action_a_holding[1]:

            if self.NoCollisions(targetx_holding, targety_holding, t):
                self.coopTransportSteps[t] += 1
                for x, y, order in zip(targetx_holding, targety_holding, orders):
                    self.moveAgent(order, x, y)
                self.moveFood(t, targetx_holding, targety_holding)
                
            # self.UpdateCapsuleMoves()   
        self.flagsDelivered(t)
        

    def moveAgent(self, id, targetx, targety):
        '''
        Moves agent to target x and y
        :param targetx: Int, target x coordinate
        :param targety: Int, target y coordinate
        '''
        
        t = id//self.c.NUMBER_OF_AGENTS_per_TEAM
        aid = id % self.c.NUMBER_OF_AGENTS_per_TEAM
        # print('agent yx: ', t, aid, self.agents_yx[t], targetx)
        self.agents_yx[t][aid] = (targety, targetx)
        

    def moveFood(self, t, targetx_holding, targety_holding):

        

        targetx = int((targetx_holding[0] + targetx_holding[1])/2)
        targety = int((targety_holding[0] + targety_holding[1])/2)
        self.s_t[self.foods_yx[1-t][0]][self.foods_yx[1-t][1]] -= self.c.FOOD
        # self.foods_yx[t][1] = targetx
        # self.foods_yx[t][0] = targety
        self.foods_yx[1-t] = (targety, targetx)
        self.s_t[self.foods_yx[1-t][0]][self.foods_yx[1-t][1]] += self.c.FOOD

    def PenaltyCollision(self, x, y, t, aid):
        '''
        Checks if x, y coordinate is currently empty 
        :param x: Int, x coordinate
        :param y: Int, y coordinate
        '''
        
        if x < 0 or x >= self.c.GW or\
        y < 0 or y >= self.c.GH or\
        self.s_t[y][x] == self.c.OBSTACLE or (y,x) in [self.agents_yx[t][j] for j in range(self.c.NUMBER_OF_AGENTS_per_TEAM) if j != aid ] or\
            (y,x) in self.agents_yx[1-t]:
            self.r[t][aid] -= 0.05
        #     return False
        # else:
        #     return True
        
    def noCollision(self, x, y, t, aid):
        '''
        Checks if x, y coordinate is currently empty 
        :param x: Int, x coordinate
        :param y: Int, y coordinate
        '''
        
        if x < 0 or x >= self.c.GW or\
        y < 0 or y >= self.c.GH or\
        self.s_t[y][x] == self.c.OBSTACLE or self.s_t[y][x] == self.c.FOOD:
            self.r[t][aid] -= 0.05
            return False
        else:
            return True
        
    def NoAgentCollision(self, x, y, id):
        #Check if x, y is currently occupied by another agent
        
        t = id//self.c.NUMBER_OF_AGENTS_per_TEAM
        aid = id % self.c.NUMBER_OF_AGENTS_per_TEAM
        if (y,x) in [self.agents_yx[t][j] for j in range(self.c.NUMBER_OF_AGENTS_per_TEAM) if j != aid ]:
            self.r[t][aid] -= 0.05
            return False
        elif (y,x) in self.agents_yx[1-t]:
            for j in range(self.c.NUMBER_OF_AGENTS_per_TEAM):
                if (y,x)== self.agents_yx[1-t][j]:
                    if (x >= self.c.MID  and t == 0) or (x < self.c.MID  and t == 1):
                        self.agents_yx[t][aid] = copy.deepcopy(self.c.AGENTS_YX[t][aid]) 
                        self.r[t][aid] -= 0.05
                        
                        return False 
                    else:
                        # self.r[t][aid] += 0.5
                        self.agents_yx[1-t][j] = copy.deepcopy(self.c.AGENTS_YX[1-t][j]) 
                        
                        self.s_t[self.foods_yx[t][0]][self.foods_yx[t][1]] -= self.c.FOOD
                        self.foods_yx[t] = copy.deepcopy(self.c.FOODS_YX[t])
                        self.s_t[self.foods_yx[t][0]][self.foods_yx[t][1]] += self.c.FOOD

                        if self.holding_flags[1-t][j]:
                            self.holding_flags[1-t][j] = False
                            for jj in range(self.c.NUMBER_OF_AGENTS_per_TEAM):
                                if jj != j and self.holding_flags[1-t][jj]:
                                    self.agents_yx[1-t][jj] = copy.deepcopy(self.c.AGENTS_YX[1-t][jj])
                                    self.holding_flags[1-t][jj] = False  
                        return True 
                            
        
        else: return True

    def NoCollisions(self, xs, ys, t):
        #Check if x, y is currently occupied by another agent
        # t = id//2 # t = 0 blue team, otherwise red team
        # aid = id  % 2 # id of agent in team t 
        if xs[0] < 0 or xs[0] >= self.c.GW or xs[1] < 0 or xs[1] >= self.c.GW or\
        ys[0] < 0 or ys[0] >= self.c.GH or ys[1] < 0 or ys[1] >= self.c.GH or\
        (ys[0],xs[0]) in self.agents_yx[1-t] or (ys[1],xs[1]) in self.agents_yx[1-t] or\
        (int(np.mean(ys)),int(np.mean(xs))) in self.agents_yx[1-t] or \
        self.s_t[ys[0]][xs[0]] == self.c.OBSTACLE or self.s_t[ys[1]][xs[1]] == self.c.OBSTACLE:
            
            if (ys[0],xs[0]) in self.agents_yx[1-t] or \
               (ys[1],xs[1]) in self.agents_yx[1-t]:
                # self.foods_yx[t] = copy.deepcopy(self.c.FOODS_XY[t])
                self.s_t[self.foods_yx[1-t][0]][self.foods_yx[1-t][1]] -= self.c.FOOD
                self.foods_yx[1-t] = copy.deepcopy(self.c.FOODS_YX[1-t])
                self.s_t[self.foods_yx[1-t][0]][self.foods_yx[1-t][1]] += self.c.FOOD
                for jj in range(2):
                    self.r[t][jj] -= 0.05
                    self.agents_yx[t][jj] = copy.deepcopy(self.c.AGENTS_YX[t][jj])
                    self.holding_flags[t][jj] = False
            return False
        else: return True

        
        

        
    
