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
import matplotlib.pyplot as plt

rom_path = './AutoROM'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
#torch.manual_seed(1)
#np.random.seed(1)

# Create the environment
env = boxing_v2.env(render_mode=None, auto_rom_install_path=rom_path)
#env = wrappers.CaptureStdoutWrapper(env)
env = wrappers.AssertOutOfBoundsWrapper(env)
env = wrappers.OrderEnforcingWrapper(env)

# Define hyperparameters
state_shape = (84, 84)
action_size = env.action_space(env.possible_agents[0]).n  # Assuming all agents have the same action space
learning_rate = 0.00025
discount_rate = 0.99
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
num_episodes = 500
update_target_network_every = 1000
replay_buffer_capacity = 1000
batch_size = 32
total_points=[]


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
        replay_buffer.buffer =training_state['replay_buffer']   # Load the buffer
        del training_state['replay_buffer']  # Remove the buffer from the state
        training_state['replay_buffer'] = replay_buffer  # Add the replay buffer object
        return training_state
    except FileNotFoundError:
        return None

def preprocess_state(state):
    #print(transform(state).shape)
    return transform(state)

class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*2*2, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
      
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Initialize the Q-network and target network for the agent
agent = env.possible_agents[0]  # Choose the first agent for training
q_network = QNetwork(action_size).float().to(device)
target_network = QNetwork(action_size).float().to(device)
target_network.load_state_dict(q_network.state_dict())

# Define the optimizer
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Helper function to get the action according to the epsilon-greedy policy
def choose_action(state, epsilon,agent12):
    if np.random.rand() <= epsilon:
        return env.action_space(agent12).sample()
    else:
        with torch.no_grad():
            stateTemp=state.unsqueeze(0)
            q_values = q_network(stateTemp)
            #print(q_values)
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
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        while True:
            batch = random.sample(self.buffer, batch_size)
        # 检查是否有 None
            if None not in batch:
                break
        
       
       # with open('myfile.txt', 'w') as file:
        #    file.write(str(batch))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
        
    def deleteNone(self):
        self.buffer=[item for item in self.buffer if item is not None]

# Load training progress if available
training_progress = load_training_progress()
if training_progress:
    q_network.load_state_dict(training_progress['q_network_state_dict'])
    target_network.load_state_dict(training_progress['target_network_state_dict'])
    epsilon = training_progress['epsilon']
    epsilon =0
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
    next_observation, _, _, _, _ = env.last()
    

    state = preprocess_state(next_observation)
    total_reward = 0
    get_points=0
    lose_points=0

    
    action = env.action_space("first_0").sample()
    while action is None:
        action = env.action_space(agent12).sample()
    for agent12 in env.agent_iter():
     
        
        
            #print("choose action first 0")
       
        next_observation, reward, done, truncation, _ = env.last()
        
        next_state = preprocess_state(next_observation)

        
            #print("player first 0")
        
 
        if agent12=='first_0':
            
            if reward>0:
                get_points+=reward
                print("Get points:",get_points)
                reward=reward*1000
            elif reward<0:
                lose_points-=reward
                print("Lose points:",lose_points)
                reward=reward*10
                
            total_reward += reward

                # Store the experience in the replay buffer
            replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            if len(replay_buffer) > batch_size:
                    #print(len(replay_buffer))
                    #print(batch_size)
                sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = replay_buffer.sample(batch_size)
                sampled_states = torch.tensor(sampled_states,device=device).float()
                sampled_actions = torch.tensor(sampled_actions,device=device).long()
                sampled_rewards = torch.tensor(sampled_rewards,device=device).float()
                sampled_next_states = torch.tensor(sampled_next_states,device=device).float()
                sampled_dones = torch.tensor(sampled_dones,device=device).float()

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
        #else:
            #print("player second 0")

        if done or truncation:
            print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            break

        if agent12=='first_0':
            action = choose_action(state, epsilon,agent12)
            state = next_state
        else:
            action = env.action_space("second_0").sample()
        
        env.step(action)
        
        step_count += 1
        if step_count % update_target_network_every == 0:
            target_network.load_state_dict(q_network.state_dict())



    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    total_points.append(get_points)
    with open('list.txt', 'w') as file:
        for item in total_points:
            file.write(f"{item}\n")
    plt.plot(total_points)
    plt.title("Points over episode")
    plt.xlabel("Episode")
    plt.ylabel("Point")
    plt.savefig("deepQlearning.png")
    plt.close()
    
    # Save training progress
    save_training_progress(q_network, target_network, epsilon, episode, replay_buffer)
    print("Progress saved, Episode:", episode)
