from pettingzoo.atari import boxing_v2
import numpy as np
from tqdm import tqdm


# there are 4 different groups in total
GROUP_NUMBER = 4
# agent number in each group
AGENT_NUMBER = 1

rom_path = './AutoROM'

env = boxing_v2.parallel_env(render_mode="human", auto_rom_install_path=rom_path)
nS=100800
nA=18
nP=2

def to1D(observation):
    value=0
    for i in range(210):
        for j in range(160):
            value=value+observation[i,j]*210*160+j*210+i;

    return value

def q_learning(num_episodes, checkpoints):
    
    Q = np.zeros((nP,nS,nA))
    num_updates = np.zeros((nP,nS,nA))
    V = np.zeros((nP,nS))

    gamma = 0.9
    epsilon = 0.9

    observation, info = env.reset()

    V_opt_checkpoint_values = []
    optimal_policy = np.zeros((nP,nS),dtype=int)


    for period in tqdm(range(num_episodes)):
        
        #actionList=[]
        observation, info = env.reset()
        while True:
            oldOBS=to1D(observation)
            actions={}
            actionListForPlayers=[]
            
            for i in range(nP):
                calculate=np.random.random()
                action=0
                if calculate>epsilon:
                    action=np.random.randint(0,nA)
               
    
                else:
               
                    currentMax=float('-inf')
                    maxOptions=[]
                    for k in range(18):
                        if Q[i][to1D(observation)][k]>currentMax:
                            currentMax=Q[i][to1D(observation)][k]
                            maxOptions=[k]
                        elif Q[i][to1D(observation)][k]==currentMax:
                            maxOptions.append(k)
                    action=maxOptions[0]
                currentAgent=env.agents[i]
                actions[currentAgent]=action
                actionListForPlayers.append(action)


            
            observation, reward, terminated, truncated, info = env.step(actions)
            
            for i in range(nP):
                eta=1/(1+num_updates[i][oldOBS][actionListForPlayers[i]])
                currentAgent=env.agents[i]
                Q[i][oldOBS][actionListForPlayers[i]]=(1-eta)*Q[i][oldOBS][actionListForPlayers[i]]+eta*(reward[currentAgent]+gamma*V[i][to1D(observation)])
                num_updates[i][oldOBS][actionListForPlayers[i]]+=1
                max_value=float('-inf')
            
                for k in range(nA):
                    if Q[i][oldOBS][k]>max_value:
                        V[i][oldOBS]=Q[i][oldOBS][k]
                        optimal_policy[i][oldOBS]=k
                        max_value=Q[i][oldOBS][k]
            if terminated or truncated:
                #print(actionList)
                break
        
        if len(checkpoints)>0:
            if checkpoints[0]-1==period:
                
                V_opt_checkpoint_values.append(V.copy())
                checkpoints.pop(0)
                

        epsilon=0.9999*epsilon
    env.close()

    return Q, optimal_policy, V_opt_checkpoint_values


def main():

    Q, optimal_policy, V_opt_checkpoint_values = q_learning(10000, checkpoints=[10,100,1000,10000])



main()
    

   

