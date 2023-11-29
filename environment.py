import re

class Environment(object):

    """ Environment """
    def __init__(self, flags):
        self.__name = flags.environment

        if 'CTF' in self.name:
            params = self.name.split('_')
            if 'V' in params[1]:
                version = int(re.sub("[^0-9]", "", params[1]))
                if version == 1:
                    
                    from env.CTF.CTFv1 import CTF
                    self.__env = CTF(version, flags.agents, deliver = flags.deliver)
                
            else:
                raise ValueError('Invalid environment string format for CMOTP')
       
        # else:
        #     from env.openai_gym.openai_gym import OpenAI_Gym
        #     self.__env = OpenAI_Gym(flags)
        #     self.__upper_bound = self.env.upper_bound
        #     self.__lower_bound = -self.env.lower_bound
        self.__fieldnames = self.__env.fieldnames
        self.__dim = self.env.dim
        self.__out = self.env.out

    def getHW(self):
        return self.__env.getHW()

    def getSaliencyCoordinates(self):
        return self.__env.getSaliencyCoordinates()

    def getAgentCoordinates(self):
        return self.__env.getAgentCoordinates()

    def processSaliency(self, saliency, folder):
        self.__env.processSaliency(saliency, folder)

    def processSaliencyCoordinates(self, saliency, folder):
        self.__env.processSaliencyCoordinates(saliency, folder)

    @property
    def upper_bound(self):
        return self.__upper_bound

    @property
    def lower_bound(self):
        return self.__lower_bound

    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    @property
    def name(self):
        return self.__name

    @property
    def env(self):
        return self.__env

    @property
    def fieldnames(self):
        return self.__fieldnames

    def evalReset(self, evalType):
        '''
        Reset for evaulations
        '''
        return self.__env.evalReset(evalType)

    def reset(self):
        '''
        Reset env to original state
        '''
        return self.__env.reset()

    def render(self):
        '''
        Render the environment
        '''
        self.__env.render()

    def step(self, a, b):
        '''
        :param float: action
        '''
        return self.__env.step(a, b)

    def stats(self):
        '''
        :return: stats from env
        '''
        return self.__env.stats()
    
    def result(self):
        '''
        :return: stats from env
        '''
        return self.__env.result()
