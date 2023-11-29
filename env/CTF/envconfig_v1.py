import random
class EnvConfigV1:

    ''' Env. V1 Parameters '''
    def __init__(self, number_agents):
        """ Gridworld Dimensions """
        self.GRID_HEIGHT = 16
        self.GRID_WIDTH = 30
        self.MID = self.GRID_WIDTH // 2
        self.GH = self.GRID_HEIGHT
        self.GW = self.GRID_WIDTH
        self.DIM = [self.GH, self.GW]
        self.HMP = int(self.GW//2) # HMP = Horizontal Mid Point
        self.VMP = int(self.GH//2) # VMP = Vertical Mid Point
        self.ACTIONS = 5 # No-op, move up, down, left, righ

        """ Wind (slippery surface) """
        self.WIND = 0.0
       
        """ Agents """
        self.NUMBER_OF_AGENTS_per_TEAM = number_agents
        self.NUMBER_OF_AGENTS = 2*self.NUMBER_OF_AGENTS_per_TEAM
        # self.AGENTS_X =[1, self.GW-1] 
        # self.AGENTS_Y = [self.GH-1, self.GH-1]
        self.AGENTS_YX = [[], []]
        # self.AGENTS_XY[0] = []
        # self.AGENTS_XY[1] = []
        ni1 = 0
        ni2 = 0
        for i in range(self.NUMBER_OF_AGENTS_per_TEAM):
            ni1 = 0
            ni2 = 0
            if i % 2 == 0:
                self.AGENTS_YX[0].append((1+ni1,1))
                self.AGENTS_YX[1].append((1+ni1,self.GW-2))
                ni1 += 1
            else:
                self.AGENTS_YX[0].append((self.GH-2-ni2,1))
                self.AGENTS_YX[1].append((self.GH-2-ni2,self.GW-2))
                ni2 += 1

        """ Goods """
        # self.GOODS_X = self.MID # Pickup X Coordinate
        # self.GOODS_Y = 11       # Pickup y Coordinate
        self.FOODS_YX = [(self.VMP, self.MID-4), (self.VMP, self.MID+3)]
      
        # """ DZONE (X, Y, Reward) """
        # self.DZONES = [(self.MID-2,0, lambda: 1.0 if random.uniform(0, 1) > 0.4 else 0.4),
        #                (self.MID+2,0, lambda: 0.8)] 
 
        """ Colors """
        self.AGENTS = [240.0, 50.0] # Colors [Agent1, Agent2]
        self.FOOD = 150.0
        self.OBSTACLE = 100.0

        # Noise related parameters.
        # Used to turn CMOTP into continous environment
        self.NOISE = 0
        self.MU = 1.0
        self.SIGMA = 0.0

        """ Obstacles """
        self.OBSTACLES_YX = []

        # Left column is made unavaialble:
        for i in range(self.GH):
            self.OBSTACLES_YX.append((i, 0))
            self.OBSTACLES_YX.append((i, self.GW-1))
        
        for i in range(self.GW):
            self.OBSTACLES_YX.append((0, i))
            self.OBSTACLES_YX.append((self.GH-1, i))

        # Wall is initialised to seperate agents from the main room: 
        for i in range(0, self.GH):
            if i != self.VMP:
                self.OBSTACLES_YX.append((i, 2))
                self.OBSTACLES_YX.append((i, self.GW-3))

        for i in range(self.VMP-2, self.VMP+3):
            self.OBSTACLES_YX.append((i, self.MID-3))
            self.OBSTACLES_YX.append((i,self.MID+2))

        for i in range(self.VMP-4, self.VMP+4):
            self.OBSTACLES_YX.append((i, self.MID-6))
            self.OBSTACLES_YX.append((i,self.MID+5))
        # Bottleneck Layers:
        #for i in range(0, self.MID-1):
        #    self.OBSTACLES_YX.append((3, i))

        #for i in range(self.MID+2, self.GW):
        #    self.OBSTACLES_YX.append((3, i))

        #for i in range(5, self.GW-4):
        #    self.OBSTACLES_YX.append((5, i))

        # for i in range(0, 6):
        #     self.OBSTACLES_YX.append((8, i))

        # for i in range(self.GW-5, self.GW):
        #     self.OBSTACLES_YX.append((8, i))

        # # Platform above the goods
        # for i in range(5, self.GW-4):
        #     self.OBSTACLES_YX.append((10, i))

        # # Touchdown area 
        # for i in range(0, self.MID - 3):
        #     self.OBSTACLES_YX.append((0, i))
        # self.OBSTACLES_YX.append((0,self.MID))
        # for i in range(self.MID + 4, self.GW):
        #     self.OBSTACLES_YX.append((0,i))

