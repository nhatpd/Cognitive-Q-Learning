# import tflearn
import math
class Config(object):
        
    def __init__(self,\
        dim,\
        out,\
        method='boltzmann',\
        mamethod=None,\
        gamma=0.95,\
        alpha=0.5,
        nframe = 1):

        # Experience Replay Memory Config
        # self.erm = self.Experience_Replay_Memory_Config() 
        self.cog = self.Cognitive_Config()
        self.eps = self.Eps_Greedy()
        self.hys = self.Hysteretic_Config()
        self.len = self.Leniency_Config()

        # MA-DRL Add-ons:
        # self.leniency = self.Leniency_Config()
        # self.hysteretic = self.Hysteretic_Config()

        # Hyperparamters 
        self.__method          = method         # IBL algorithm.
        self.__mamethod        = mamethod        # MA-IBL algorithms
        self.__outputs      = out          # Number of outputs
        self.__gamma        = gamma        # Discount rate
        self.__dim          = dim          # Set input dimensions
        self.__id           = None
        self.__alpha        = alpha        #learning rate of TD
        self.__nframe       = nframe       # number of frames stacked

    def __repr__(self):
        return str(vars(self))
    
    @property
    def nframe(self):
        return self.__nframe
    @property
    def alpha(self):
        return self.__alpha

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, value):
        if self.__method == None:
            self.__method = value
        else:
            raise Exception("Can't modify method.")

    @property
    def mamethod(self):
        return self.__mamethod

    @mamethod.setter
    def mamethod(self, value):
        if self.__mamethod == None:
            self.__mamethod = value
        else:
            raise Exception("Can't modify method.")

    @property
    def outputs(self):
        return self.__outputs

    @property
    def gamma(self):
        return self.__gamma

    @property
    def dim(self):
        return self.__dim

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        self.__id = value

    class Cognitive_Config(object):

        """ IBL Config """
        def __init__(self):
            self.__default_utility = 1.0
            self.__noise = 0.25
            self.__decay = 0.5
            self.__temperature = 0.25*math.sqrt(2)
            self.max_references = 5000 #1000
        def __repr__(self):
            return str(vars(self))

        @property
        def default_utility(self):
            return self.__default_utility
            
        @default_utility.setter
        def default_utility(self, value):
            self.__default_utility = value
        
        @property
        def noise(self):
            return self.__noise
        @noise.setter
        def noise(self, value):
            self.__noise = value
        
        @property
        def decay(self):
            return self.__decay
        @decay.setter
        def decay(self, value):
            self.__decay = value
        
        @property
        def temperature(self):
            return self.__temperature
        @temperature.setter
        def temperature(self, value):
            self.__temperature = value 
        
        @property
        def max_references(self):
            return self.__max_references
        @max_references.setter
        def max_references(self, value):
            self.__max_references = value

    class Eps_Greedy(object):

        """ Epsilon-Greedy Exploration """
        def __init__(self):
            self.__initial  = 1.0
            self.__min      = 0.1
            self.__update   = 1      # Update eps every n episodes
            self.__discount = 0.999  # Epsilong discount factor old 0.995

        def __repr__(self):
            return str(vars(self))

        @property
        def initial(self):
            return self.__initial

        @property
        def min(self):
            return self.__min

        @property
        def update(self):
            return self.__update

        @property
        def discount(self):
            return self.__discount

    class Hysteretic_Config(object):

        """ Hysteretic Hyperparameters """
        def __init__(self):
            self.__beta = 0.5*0.001 # Percentage of alpha - second learning rate of hysterestic

        def __repr__(self):
            return str(vars(self))

        @property
        def beta(self):
            return self.__beta

        @beta.setter
        def beta(self, value):
            self.__beta = value


    class Leniency_Config(object):

        """ Leniency related hyperparameters """
        def __init__(self, method='TDS'):
            self.__ase                   = 0.25         # Action Selection Exponent
            self.__max                   = 1.0          # Initial temperature
            self.__min                   = 0.0          # Min temperature value
            self.__tmc                   = 1.0          # Leniency moderatoin coefficient K = theta
            self.__hashing               = 'xxhash'     # Options: xxhash, AutoEncoder, L2 (Layer2)
            self.__max_temperature_decay = 0.9998       # Used for global temperature decay
            self.__threshold             = 200000       # Temperature update threshold
            # if method == 'TDS':
            #     self.__method = self.Temperature_Decay_Schedule()
            # elif method == 'ATF':
            #     self.__method = self.Average_Temperature_Folding()
            # AutoEncoder related parameters
            self.__aestart      = 50000                 # Steps after which optimsiation of the ae starts
            self.__aeend        = 250000                # Steps after which the ae is no longer trained

            self.__delta = 0.999 # old 0.995
            self.__theta = 1 #old 1 lenience moderation factor
            self.__tau = 0.1

        def __repr__(self):
            return str(vars(self))

        @property
        def delta(self):
            return self.__delta
        
        @property
        def theta(self):
            return self.__theta
        
        @property
        def tau(self):
            return self.__tau
        
        @property
        def aestart(self):
            return self.__aestart

        @property
        def aeend(self):
            return self.__aeend

        @property
        def ase(self):
            return self.__ase

        @property
        def max(self):
            return self.__max

        @property
        def min(self):
            return self.__min

        @property
        def tmc(self):
            return self.__tmc

        @property
        def hashing(self):
            return self.__hashing

        @property
        def max_temperature_decay(self):
            return self.__max_temperature_decay

        @property
        def method(self):
            return self.__method

        @property
        def threshold(self):
            return self.__threshold
        
        @property
        def ase(self):
            return self.__ase

