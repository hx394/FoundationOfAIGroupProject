

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

# Create the environment
rom_path = './AutoROM'






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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

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
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
memory_size = 100000
batch_size = 64
num_episodes = 1000

# Environment setup
env = boxing_v2.parallel_env(render_mode="human",auto_rom_install_path=rom_path)
num_agents = 2  # Assuming we have 2 agents
action_size = env.action_space(env.possible_agents[0]).n

# Initialize networks and memory for each agent
q_networks = MultiAgentQNetwork(action_size, num_agents).float()
target_networks = MultiAgentQNetwork(action_size, num_agents).float()
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

# Training loop for multiple agents
for episode in range(num_episodes):
    observations_tuple = env.reset()
    observations = observations_tuple[0]  # Extract the observations from the first element of the tuple
    states = {agent: preprocess_state(observations[agent]) for agent in env.agents}
    total_rewards = {agent: 0 for agent in env.agents}

    while True:
        actions = {}
        for agent_index, agent in enumerate(env.agents):
            action = choose_action(states[agent], epsilon)
            actions[agent] = action

        next_observations_tuple, rewards, dones, truncations, infos = env.step(actions)
        next_observations = next_observations_tuple  # Extract the observations from the first element of the tuple
        next_states = {agent: preprocess_state(next_observations[agent]) for agent in env.agents}

        for agent_index, agent in enumerate(env.agents):
            memory = memories[agent_index]
            state = states[agent]
            action = actions[agent]
            reward = rewards[agent]
            next_state = next_states[agent]
            done = dones[agent] or truncations[agent]
            total_rewards[agent] += reward

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
                with torch.no_grad():
                    next_q_values = target_networks(next_states_batch, agent_index).max(dim=1)[0]
                    targets = rewards_batch + gamma * next_q_values * (~dones_batch)

                # Compute current Q-values
                q_values = q_networks(states_batch, agent_index).gather(1, actions_batch.unsqueeze(1)).squeeze()

                # Update the Q-network
                optimizer = optim.Adam(q_networks.networks[agent_index].parameters(), lr=learning_rate)
                loss = nn.functional.smooth_l1_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        states = next_states

        if all(dones.values()) or all(truncations.values()):
            break

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update the target network
    if episode % 10 == 0:
        target_networks.load_state_dict(q_networks.state_dict())

    print(f"Episode: {episode + 1}, Total Rewards: {total_rewards}")

env.close()

