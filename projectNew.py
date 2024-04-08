import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import transforms
from pettingzoo.atari import boxing_v2
from pettingzoo.utils import wrappers
import pickle

rom_path = './AutoROM'

# Set random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Create the environment
env = boxing_v2.env(render_mode="human", auto_rom_install_path=rom_path)
env = wrappers.CaptureStdoutWrapper(env)
env = wrappers.AssertOutOfBoundsWrapper(env)
env = wrappers.OrderEnforcingWrapper(env)

# Define hyperparameters
state_shape = (84, 84)
action_size = env.action_space(env.possible_agents[0]).n  # Assuming all agents have the same action space
learning_rate = 0.00025
discount_rate = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000
update_target_network_every = 1000
replay_buffer_capacity = 10000
batch_size = 32

# Preprocessing function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize(state_shape),
    transforms.ToTensor()
])

def save_training_progress(q_network, target_network, epsilon, episode, replay_buffer, filename='training_progress.pkl'):
    training_state = {
        'q_network_state_dict': q_network.state_dict(),
        'target_network_state_dict': target_network.state_dict(),
        'epsilon': epsilon,
        'episode': episode,
        'replay_buffer': replay_buffer.buffer  # Save the buffer
    }
    with open(filename, 'wb') as f:
        pickle.dump(training_state, f)

def load_training_progress(filename='training_progress.pkl'):
    try:
        with open(filename, 'rb') as f:
            training_state = pickle.load(f)
        replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        replay_buffer.buffer = training_state['replay_buffer']  # Load the buffer
        del training_state['replay_buffer']  # Remove the buffer from the state
        training_state['replay_buffer'] = replay_buffer  # Add the replay buffer object
        return training_state
    except FileNotFoundError:
        return None

def preprocess_state(state):
    return transform(state).unsqueeze(0)

# Define the Q-network with convolutional layers
class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(9*9*32, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the Q-network and target network for the agent
agent = env.possible_agents[0]  # Choose the first agent for training
q_network = QNetwork(action_size).float()
target_network = QNetwork(action_size).float()
target_network.load_state_dict(q_network.state_dict())

# Define the optimizer
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Helper function to get the action according to the epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space(agent).sample()
    else:
        with torch.no_grad():
            q_values = q_network(state)
       
        return q_values.argmax().item()

# Define the ReplayBuffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        state = np.squeeze(state, axis=0)
        next_state = np.squeeze(next_state, axis=0)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Load training progress if available
training_progress = load_training_progress()
if training_progress:
    q_network.load_state_dict(training_progress['q_network_state_dict'])
    target_network.load_state_dict(training_progress['target_network_state_dict'])
    epsilon = training_progress['epsilon']
    start_episode = training_progress['episode'] + 1
    replay_buffer = training_progress['replay_buffer']  # Load the replay buffer
    print("Successfully loaded progress")
else:
    start_episode = 0
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)  # Initialize a new replay buffer
    print("Loading progress failed")

# Training loop
step_count = 0
for episode in range(start_episode, num_episodes):
    env.reset()
    observation, _, _, _, _ = env.last()
    state = preprocess_state(observation)
    total_reward = 0
    get_points=0
    lose_points=0
    for agent12 in env.agent_iter():
        if agent12 == 'second_0':
            action = env.action_space(agent12).sample()
        else:
            action = choose_action(state, epsilon)
        env.step(action)
        next_observation, reward, done, truncation, _ = env.last()

        if agent12 == 'first_0':
            next_state = preprocess_state(next_observation)
            
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
            total_reward += reward

            # Store the experience in the replay buffer
            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            if len(replay_buffer) > batch_size:
                sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = replay_buffer.sample(batch_size)
                sampled_states = torch.tensor(sampled_states).float()
                sampled_actions = torch.tensor(sampled_actions).long()
                sampled_rewards = torch.tensor(sampled_rewards).float()
                sampled_next_states = torch.tensor(sampled_next_states).float()
                sampled_dones = torch.tensor(sampled_dones).float()

                # Compute the target Q-values
                with torch.no_grad():
                    max_next_q_values = target_network(sampled_next_states).max(1)[0]
                    target_q_values = sampled_rewards + discount_rate * max_next_q_values * (1 - sampled_dones)

                # Compute the current Q-values
                current_q_values = q_network(sampled_states).gather(1, sampled_actions.unsqueeze(1)).squeeze(1)

                # Update the Q-network
                loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            step_count += 1
            if step_count % update_target_network_every == 0:
                target_network.load_state_dict(q_network.state_dict())

        if done or truncation:
            print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            break

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Save training progress
    save_training_progress(q_network, target_network, epsilon, episode, replay_buffer)
    print("Progress saved, Episode:", episode)
