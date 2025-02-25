import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

TF_ENABLE_ONEDNN_OPTS=0
# Reward shaping function
def reward_shaping(state, reward, done):
    if done:
        return -10  # Penalty if the episode ends (failed)
    return reward + abs(state[0])  # Additional reward based on the state


# Custom wrapper for reward shaping
class RewardShapingWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super(RewardShapingWrapper, self).__init__(venv)

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.actions = actions
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Apply reward shaping to each environment
        for i in range(len(rewards)):
            rewards[i] = reward_shaping(obs[i], rewards[i], dones[i])
        return obs, rewards, dones, infos


# Custom callback to collect training metrics
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.rewards = []
        self.losses = []
        self.episode_lengths = []
        self.actions_taken = []  # Store actions taken

    def _on_step(self) -> bool:
        # Collect rewards
        self.rewards.append(self.locals['rewards'][0])

        # Collect losses (if available)
        if 'loss' in self.locals:
            self.losses.append(self.locals['loss'])

        # Collect episode lengths
        if 'episode' in self.locals:
            self.episode_lengths.append(self.locals['episode']['l'][0])

        # Collect actions taken
        if 'actions' in self.locals:
            self.actions_taken.extend(self.locals['actions'])  # Store actions

        return True


# Hyperparameters
n_envs = 4  # Parallel environments
total_timesteps = 500_000  # Train longer than PPO

# Create vectorized + normalized environment
env = make_vec_env('Acrobot-v1', n_envs=n_envs)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Wrap the environment with reward shaping
env = RewardShapingWrapper(env)

# Network architecture (simpler than PPO)
policy_kwargs = dict(net_arch=[64, 64])

# Initialize A2C model
model = A2C(
    policy="MlpPolicy",
    env=env,
    learning_rate=7e-4,  # Higher than PPO
    n_steps=256,  # Shower updates
    gamma=0.99,
    ent_coef=0.01,  # Balanced exploration
    max_grad_norm=0.5,  # Prevent exploding gradients
    policy_kwargs=policy_kwargs,
    verbose=1,
    device='cpu',
    tensorboard_log="./a2c_acrobot_tensorboard1/"
)

# Callbacks for evaluation + early stopping
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-50, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    best_model_save_path="./best_a2c_model2",
    eval_freq=5000,  # Evaluate every 5k steps
    deterministic=True,
)

# Custom metrics callback
metrics_callback = MetricsCallback()

# Train the agent
model.learn(total_timesteps=total_timesteps, callback=[eval_callback, metrics_callback], tb_log_name="a2c_acrobot")

# Save the final model
model.save("a2c_acrobot_final")

# Save the normalization statistics
env.save("env_stats.pkl")


# Plotting the metrics
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


plt.figure(figsize=(18, 10))

# Reward over Episodes
plt.subplot(2, 3, 1)
plt.plot(metrics_callback.rewards, label="Total Reward per Episode", alpha=0.5)
avg_rewards = moving_average(metrics_callback.rewards, window_size=50)
plt.plot(range(len(avg_rewards)), avg_rewards, label="Moving Avg (50 episodes)", color="red")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Reward over Episodes")
plt.legend()

# Loss over Episodes
plt.subplot(2, 3, 2)
plt.plot(metrics_callback.losses, label="Loss", color="blue")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Episode Length over Episodes
plt.subplot(2, 3, 3)
plt.plot(metrics_callback.episode_lengths, label="Episode Length", color="green")
plt.xlabel("Episodes")
plt.ylabel("Episode Length")
plt.title("Episode Length over Episodes")
plt.legend()

# TD Error (Assuming loss is close to TD error)
plt.subplot(2, 3, 4)
td_errors = np.abs(np.array(metrics_callback.losses))  # Assuming loss is close to TD error
plt.plot(td_errors, label="TD Error", color="orange")
plt.xlabel("Episodes")
plt.ylabel("Error Magnitude")
plt.title("Temporal Difference (TD) Error")
plt.legend()

# Action selection histogram
plt.subplot(2, 3, 5)
plt.hist(metrics_callback.actions_taken, bins=env.action_space.n, rwidth=0.8, align='mid', color='purple', alpha=0.7)
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.title("Action Selection Distribution")

plt.tight_layout()
plt.show()

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    env.render()

    # Check if any environment is done
    if done.any():
        # Reset only the environments that are done
        obs[done] = env.reset()[done]

env.close()