
from environment import Environment
from stats import mkRunDir
from config import Config
from configibl import Config as Configibl
from copy import deepcopy
from itertools import count
import random as random
import numpy as np 
import time
import csv
from collections import deque



from cogMAQL import cogAgentQL #Cognitive Q-Learning algorithms
from MAQL import AgentQL #Q-Learning algorithms

import argparse
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="CTF")

flags.add_argument('--environment',type=str,default='CTF_V1',help='Environment.') 
flags.add_argument('--blue_team',type=str,default='cogQL_none',help='method for blue team: Options: cogQL_none, cogQL_leniency, cogQL_hysteretic, QL__leniency, QL_hysteretic, QL_none')
flags.add_argument('--red_team',type=str,default= 'QL_none',help='method for red team: Options: None, leniency, hysteretic, ppo, ibl_leniency, ibl_hysteretic, ibl_greedy')
flags.add_argument('--agents',type=int,default=2,help='Number of agents per team.')
flags.add_argument('--episodes',type=int,default=500,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=10000,help='Number of steps.')
flags.add_argument('--start_t',type=int,default=0,help='Number of trials.')
flags.add_argument('--trials',type=int,default=20,help='Number of trials.')
flags.add_argument('--nframe',type=int,default=1,help='Number of frames stacked.')

flags.add_argument('--render',default=True,help='visualize')
flags.add_argument('--deliver',default=False,help='agents must pickup and then deliver foods')

FLAGS = flags.parse_args()


for runid in range(FLAGS.start_t,FLAGS.trials):
    
    env = Environment(FLAGS) 

    statscsv, folder = mkRunDir(env, FLAGS, runid)
    
    agents = []
    # Create first team: cognitive QL
    for i in range(FLAGS.agents): 

        if FLAGS.blue_team.split('_')[0] == 'cogQL':
            configibl = Configibl(env.dim, env.out, mamethod=FLAGS.blue_team.split('_')[1], nframe = FLAGS.nframe)
            agent_configibl = deepcopy(configibl)

            agents.append(cogAgentQL(agent_configibl))
            
        elif FLAGS.blue_team.split('_')[0] == 'QL':
            configibl = Configibl(env.dim, env.out, mamethod=FLAGS.blue_team.split('_')[1], nframe = FLAGS.nframe)
            agent_configibl = deepcopy(configibl)

            agents.append(AgentQL(agent_configibl))

    # Create second team: QL
    for i in range(FLAGS.agents): 

        if FLAGS.red_team.split('_')[0] == 'cogQL':
            configibl = Configibl(env.dim, env.out, mamethod=FLAGS.red_team.split('_')[1], nframe = FLAGS.nframe)
            agent_configibl = deepcopy(configibl)

            agents.append(cogAgentQL(agent_configibl))
            
        elif FLAGS.red_team.split('_')[0] == 'QL':
            configibl = Configibl(env.dim, env.out, mamethod=FLAGS.red_team.split('_')[1], nframe = FLAGS.nframe)
            agent_configibl = deepcopy(configibl)

            agents.append(AgentQL(agent_configibl))
    ##################

    # Start training run
    ord_e = random.sample(range(2*FLAGS.agents),1)[0]
    fixed_id_team = ord_e // FLAGS.agents
    for i in range(FLAGS.episodes):
        
        # Run episode
        observations = env.reset() # Get first observations
        
        start = time.time()
        
        
        for j in range(FLAGS.steps):
            #######################################
            # Renders environment if flag is true
            if FLAGS.render: 
                env.render() 
                
            
            r = [[0 for i in range(FLAGS.agents)], [0 for i in range(FLAGS.agents)]]
            
            if FLAGS.nframe == 2*FLAGS.agents:
                obs = []
                for jjj in range(2):
                    obs.append([deque([deepcopy(observations[jjj][i]) for jj in range(FLAGS.nframe)],FLAGS.nframe) for i in range(FLAGS.agents)])
            actions = [[],[]]
            orders = [[],[]]

            for ord in range(2*FLAGS.agents):
                
                ordx = ord_e % (2*FLAGS.agents)
                id_team = ordx // FLAGS.agents
                id_a = ordx % FLAGS.agents
                
                if FLAGS.nframe == 2*FLAGS.agents:
                    o = np.array(obs[id_team][id_a])
                else:
                    o = observations[id_team][id_a]

                actions[id_team].append(agents[ordx].move(o))
                orders[id_team].append(ordx)
                
                

                ord_e += 1
            
            observations, r, t = env.step([actions[fixed_id_team], actions[1-fixed_id_team]], [orders[fixed_id_team],orders[1-fixed_id_team]])
                
            for ord in range(2*FLAGS.agents):
            
                
                if FLAGS.nframe == 2*FLAGS.agents:
                    o = np.array(obs[ord//FLAGS.agents][ord % FLAGS.agents])
                else:

                    o = observations[ord//FLAGS.agents][ord % FLAGS.agents]

                agents[ord].feedback(r[ord//FLAGS.agents][ord % FLAGS.agents], t, o) 

                agents[ord].opt()
            
            if t: 
                print('finished game')
                break # If t then terminal state has been reached
        end = time.time()
        env.env.time += end - start
        # Add row to stats: 
        with open(statscsv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
            writer.writerow(env.stats())
        print(env.stats())
    # with open(str(runid)+'transition.json', 'w') as testfile:
    #     json.dump(transitions,testfile)

