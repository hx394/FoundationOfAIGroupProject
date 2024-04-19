import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torchvision import transforms
from pettingzoo.atari import boxing_v2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Create the environment
rom_path = './AutoROM'
#device= torch.device("cuda" if torch.cuda.is_available else "cpu")

# Preprocessing function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()
])

def preprocess_state(state):
    return transform(state).unsqueeze(0)

# Define the Q-network with convolutional layers

class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # 输出尺寸: (84-8)/4 + 1 = 20
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)      # 输出尺寸: 20/2 = 10
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 输出尺寸: (10-4)/2 + 1 = 4
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)      # 输出尺寸: 4/2 = 2
        self.fc1 = nn.Linear(2*2*64, 48)  # 根据最终的特征图尺寸调整全连接层的输入尺寸
        self.fc2_q_values = nn.Linear(48, action_size)

        # Additional fully connected layer for exploration metric predictions
        self.fc2_exploration = nn.Linear(48, action_size)  # For exploration metric
        self.exploration_activation = nn.ReLU() # active

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)

        # Separate paths for Q-values and exploration metric
        x = torch.relu(self.fc1(x))
        q_values = self.fc2_q_values(x)
        exploration_metric = self.exploration_activation(self.fc2_exploration(x))

        return q_values, exploration_metric

# Define a Multi-Agent Q-Network
class MultiAgentQNetwork(nn.Module):
    def __init__(self, action_size, num_agents):
        super(MultiAgentQNetwork, self).__init__()
        self.num_agents = num_agents
        self.networks = nn.ModuleList([QNetwork(action_size) for _ in range(num_agents)])

    def forward(self, x, agent_index):
        return self.networks[agent_index](x)


# Hyperparameters
state_size = (84, 84)
learning_rate = 0.00025
gamma = 0.99
memory_size = 100000
batch_size = 64
num_episodes = 500
total_points = []
discount_rate = 0.99

# Environment setup
env = boxing_v2.parallel_env(render_mode="human",auto_rom_install_path=rom_path)
num_agents = 2  # Assuming we have 2 agents
action_size = env.action_space(env.possible_agents[0]).n

# UCB parameters
UCB_C = 1.0
# counts and rewards for UCB
#action_counts = np.zeros((84, num_agents, action_size))
#total_rewards = np.zeros((84, num_agents, action_size))

# Initialize networks and memory for each agent
q_networks = MultiAgentQNetwork(action_size, num_agents).float()
target_networks = MultiAgentQNetwork(action_size, num_agents).float()
try:
    q_networks.load_state_dict(torch.load('q_networks.pth'))
except FileNotFoundError:
    print("FileNotFoundError！")
except RuntimeError as e:
    print(f"RuntimeError：{e}")
target_networks.load_state_dict(q_networks.state_dict())
memories = [deque(maxlen=memory_size) for _ in range(num_agents)]

# Helper function to choose an action based on epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space(env.possible_agents[0]).sample()
    else:
        with torch.no_grad():
            q_values = q_networks(state, 0)  # Using the first network for action selection
        return q_values.argmax().item()

# Helper function to choose an action based on UCB policy
def choose_ucb_action(state, agent_index):
    """
    agent_action_counts = action_counts[agent_index, :]

    average_rewards = np.divide(total_rewards[agent_index, :], agent_action_counts,
                            out=np.zeros_like(total_rewards[agent_index, :]),
                            where=agent_action_counts!=0)

    # avoid using 0 as divisor
    safe_counts = np.where(agent_action_counts == 0, 1e-1, agent_action_counts)

    total_counts = np.sum(safe_counts)
    divisor = np.log(total_counts) / safe_counts
    divisor = np.maximum(divisor, 0)
    # conmbine of q_values and UCB
    ucb_values = q_values + average_rewards + UCB_C * np.sqrt(divisor)
    """
    q_values, exploration_metric = q_networks(state, 0)  # model output
    q_values = q_values.detach().cpu().numpy()
    exploration_metric = exploration_metric.detach().cpu().numpy()
    # get UCB_value
    ucb_values = q_values + UCB_C * np.sqrt(exploration_metric)
    action = np.argmax(ucb_values)

    return action

# Training loop for multiple agents
for episode in range(num_episodes):
    observations_tuple = env.reset()
    observations = observations_tuple[0]  # Extract the observations from the first element of the tuple
    states = {agent: preprocess_state(observations[agent]) for agent in env.agents}

    episode_reward = 0
    get_points=0
    lose_points=0

    optimizer = optim.Adam(q_networks.networks[0].parameters(), lr=learning_rate)

    while True:
        actions = {}
        for agent_index, agent in enumerate(env.agents):
            # action = choose_action(states[agent], epsilon)
            if agent_index == 0:
                action = choose_ucb_action(states[agent], agent_index)
            else:
                action = env.action_space(env.possible_agents[0]).sample()
            actions[agent] = action


        next_observations_tuple, rewards, dones, truncations, infos = env.step(actions)
        next_observations = next_observations_tuple  # Extract the observations from the first element of the tuple
        next_states = {agent: preprocess_state(next_observations[agent]) for agent in env.agents}

        for agent_index, agent in enumerate(env.agents):
            if agent_index == 1:
                continue

            # update action counts and rewards in UCB policy
            # action_counts[agent_index, action] += 1
            # total_rewards[agent_index, action] += rewards[agent]

            reward = rewards[agent]
            episode_reward += reward
            if reward>0:
                get_points+=reward
                print("Get points:",get_points)
                reward=reward*1000
            elif reward==0:
                reward=-1
            elif reward<0:
                lose_points-=reward
                print("Lose points:",lose_points)
                reward=reward*10

            memory = memories[agent_index]
            state = states[agent]
            action = actions[agent]
            reward = rewards[agent]
            next_state = next_states[agent]
            done = dones[agent] or truncations[agent]

            # Store the experience in memory
            memory.append((state, action, reward, next_state, done))

            # Experience replay
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)
                states_batch = torch.cat(states_batch)
                actions_batch = torch.tensor(actions_batch)
                rewards_batch = torch.tensor(rewards_batch)
                next_states_batch = torch.cat(next_states_batch)
                dones_batch = torch.tensor(dones_batch)

                # Compute target Q-values
                # Compute current Q-values
                current_q_values, current_exploration_metric = q_networks(states_batch, agent_index)
                next_q_values, _ = target_networks(next_states_batch, agent_index)

                max_next_q_values = next_q_values.max(dim=1)[0]
                targets_q_values = rewards_batch + discount_rate * max_next_q_values.detach() * (~dones_batch)

                baseline_exploration_metric = torch.mean(current_exploration_metric)
                exploration_metric_loss = nn.functional.mse_loss(current_exploration_metric, baseline_exploration_metric.expand_as(current_exploration_metric))


                # Update the Q-network
                q_values_loss = nn.functional.smooth_l1_loss(current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(), targets_q_values)
                total_loss = q_values_loss + exploration_metric_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        states = next_states

        if all(dones.values()) or all(truncations.values()):
            break

    total_points.append(episode_reward)
    print("episode",episode,"reward:",episode_reward)

    # Update the target network
    if episode % 10 == 0:
        target_networks.load_state_dict(q_networks.state_dict())

    print(f"Episode: {episode + 1}")
    torch.save(q_networks.state_dict(), 'q_networks.pth')

    # Save plot
    plt.plot(total_points)
    plt.title("Points over Episode")
    plt.xlabel("Epsiode")
    plt.ylabel("Point")
    plt.savefig("epsilon_chart2.png")
    plt.close()

env.close()
