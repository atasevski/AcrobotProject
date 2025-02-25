import gym
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pygame
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dqn import *


# HYPERPARAMETERS
EPISODES = 800  # Number of episodes
GAMMA = 0.99 # Discount factor
LR = 0.001 # Learning rate
EPSILON = 1.0 # Initial exploration rate
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.05
BATCH_SIZE = 64
MEMORY_SIZE = 20000

# Линеарно зголемување на бета за подобро балансирање
BETA_START = 0.4
BETA_END = 1.0
BETA_FRAMES = EPISODES

def beta_by_frame(frame_idx):
    return BETA_START + (BETA_END - BETA_START) * (frame_idx / BETA_FRAMES)


#Adding reward_reshape
def reward_shaping(state, reward, done):
    if done:
        return -10  # Ако не успее, да добие голема казна
    return reward + abs(state[0])  # Дава дополнителна награда ако е поблиску до целта

# Select Action (Epsilon-Greedy)
def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(next(policy_net.parameters()).device)
        with torch.no_grad():
            return policy_net(state).argmax().item()

def soft_update(target, source, tau=0.01):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def moving_average(values, window=50):
    return np.convolve(values, np.ones(window)/window, mode='valid')


# Training Loop
def train_dqn():
    env = gym.make("Acrobot-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).cuda()
    target_net = DQN(state_dim, action_dim).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    #memory = ReplayBuffer(MEMORY_SIZE)
    memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=0.7)

    epsilon = EPSILON
    rewards_history = []
    loss_history = []
    epsilon_history = []
    steps_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0 # for computing avg reward
        done = False
        episode_loss = []

        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward_shaping(state, reward, done) # here I reshape the reward - helping the agent to learn faster
            memory.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            if len(memory) > BATCH_SIZE:

                #batch = memory.sample(BATCH_SIZE)
                # динамично пресметување на бета
                beta = beta_by_frame(episode)
                batch, indices, weights = memory.sample(BATCH_SIZE, beta)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32 , device="cuda")
                actions = torch.LongTensor(actions).unsqueeze(1).cuda()
                rewards = torch.FloatTensor(rewards).cuda()
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device="cuda")
                dones = torch.FloatTensor(dones).cuda()

                q_values = policy_net(states).gather(1, actions).squeeze()
                # ---- THIS IS FOR DQN ----
                next_q_values = target_net(next_states).max(1)[0]

                # ---- THIS IS DDQN IMPLEMENTATION ----
                #next_actions = policy_net(next_states).argmax(1, keepdim=True)
                #next_q_values = target_net(next_states).gather(1, next_actions).squeeze()

                target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

                #loss = nn.MSELoss()(q_values, target_q_values.detach())
                loss = nn.SmoothL1Loss()(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss.append(loss.item())

                errors = (target_q_values - q_values).detach().abs().cpu().numpy()
                memory.update_priorities(indices, errors)

        if episode % 30 == 0:
            soft_update(target_net, policy_net, tau=0.05)
            #target_net.load_state_dict(policy_net.state_dict())

        #epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        epsilon = EPSILON_MIN + 0.5 * (EPSILON - EPSILON_MIN) * (1 + np.cos(np.pi * episode / EPISODES))
        rewards_history.append(total_reward)
        epsilon_history.append(epsilon)
        steps_history.append(step_count)
        loss_history.append(np.mean(episode_loss) if episode_loss else 0)
        avg_reward_per_step = total_reward / step_count if step_count > 0 else float('-inf')
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.3f}, Avg Reward per Step: {avg_reward_per_step:.3f}, Epsilon: {epsilon:.3f}")


    window_size = 50
    avg_rewards = moving_average(rewards_history, window_size)
    max_rewards = [max(rewards_history[:i + 1]) for i in range(len(rewards_history))]

    plt.figure(figsize=(18, 10))

    # Reward over Episodes (Existing)
    plt.subplot(2, 3, 1)
    plt.plot(rewards_history, label="Total Reward per Episode", alpha=0.5)
    plt.plot(range(len(avg_rewards)), avg_rewards, label=f"Moving Avg ({window_size} episodes)", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Reward over Episodes")
    plt.legend()

    # Maximum Reward per Episode
    plt.subplot(2, 3, 2)
    plt.plot(max_rewards, label="Max Reward so far", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Max Reward")
    plt.title("Best Performance Over Time")
    plt.legend()

    # Epsilon Decay (Existing)
    plt.subplot(2, 3, 3)
    plt.plot(epsilon_history)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")

    # Loss (Existing)
    plt.subplot(2, 3, 4)
    plt.plot(loss_history)
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    # Steps
    plt.subplot(2, 3, 5)
    plt.plot(steps_history)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Number of steps to reach the target")

    # Action selection histogram (Optional)
    plt.subplot(2, 3, 6)
    actions_taken = [action for (state, action, reward, next_state, done) in memory.buffer]
    plt.hist(actions_taken, bins=env.action_space.n, rwidth=0.8, align='mid', color='purple', alpha=0.7)
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.title("Action Selection Distribution")

    plt.tight_layout()
    plt.show()

    avg_reward = np.mean(rewards_history)
    avg_steps = np.mean(steps_history)
    print(f"Average Reward: {avg_reward:.3f}, Average Steps to Goal: {avg_steps:.3f}")


if __name__ == '__main__':
    train_dqn()