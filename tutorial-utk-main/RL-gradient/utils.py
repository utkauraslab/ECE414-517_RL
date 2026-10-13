# Imports
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
import seaborn as sns
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym

# making plots pretty
sns.set_palette("deep")
rc('animation', html='jshtml')
import warnings
warnings.filterwarnings('ignore')


#@title Shared helper functions (run me)

# A dictionary that accumulates the training curve of every experiment,
# so that later sections can compare against earlier ones.
all_results = {}

# A fixed colour for each method, so every comparison plot is consistent.
METHOD_COLORS = {
    'REINFORCE (linear)':   'tab:blue',
    'REINFORCE (neural)':   'tab:orange',
    'REINFORCE + baseline': 'tab:green',
    'Actor-Critic':         'tab:red',
    'PPO (Stable-Baselines3)': 'tab:purple',
}


def seed_everything(seed):
    """Set the random seeds so that a training run is reproducible.

    Running the same cell twice with the same seed produces the same learning curve.
    The environment resets draw their own seeds from numpy below, so seeding numpy
    here is enough to make the collected trajectories reproducible too.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(env, policy, T, gamma=1., num_episodes=10):
    """Estimate the policy's performance by averaging the return over several episodes.

    :param env: the Gym environment to evaluate in.
    :param policy: a policy exposing a `draw_action` method.
    :param gamma: the discount factor applied to the rewards.
    :param num_episodes: how many episodes to average over.
    :return: the mean return and its standard error across the episodes.
    """
    all_episode_rewards = []
    for _ in range(num_episodes):
        episode_rewards = []
        done = False
        discounter = 1.
        obs = env.reset(seed=int(np.random.randint(2**31 - 1)))[0]
        t = 0
        while not done and t < T:
            action = policy.draw_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward * discounter)
            discounter *= gamma
            t += 1
        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = (
        np.std(all_episode_rewards, ddof=1) / np.sqrt(num_episodes)
        if num_episodes > 1 else 0.
    )
    print("Mean reward:", round(mean_episode_reward, 2),
          "Std reward:", round(std_episode_reward, 2),
          "Num episodes:", num_episodes)
    return mean_episode_reward, std_episode_reward


def record_episode(env, policy):
    """Run a single episode and return the rendered frames, for animation."""
    frames = []
    obs = env.reset(seed=int(np.random.randint(2**31 - 1)))[0]
    done = False
    while not done:
        action = policy.draw_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
    return frames


def plot_results(results, label=None):
    """Plot the average return at each training iteration, with a shaded standard error."""
    _mean = np.array([m for m, _ in results])
    _std = np.array([s for _, s in results])
    iterations = np.arange(len(_mean))
    color = METHOD_COLORS.get(label)
    plt.plot(iterations, _mean, label=label, color=color)
    plt.fill_between(iterations, _mean - _std, _mean + _std, alpha=.2, color=color)
    plt.xlabel('Training iteration')
    plt.ylabel('Average return')
    if label:
        plt.legend(loc='lower right')


def compare_results(curves, title='Comparison'):
    """Overlay several named training curves on a single plot.

    :param curves: a dict mapping each method's name to its list of (mean, std) results.
    """
    plt.figure()
    for label, results in curves.items():
        plot_results(results, label=label)
    plt.title(title)
    plt.show()


def collect_rollouts(env, policy, m, T):
    """Run the policy in the environment to gather training data.

    :param m: the number of trajectories (rollouts) to collect.
    :param T: the maximum number of steps per trajectory.
    :return: a list of m trajectories, each a list of (state, action, reward) tuples.
    """
    trajectories = []
    for _ in range(m):
        s, _ = env.reset(seed=int(np.random.randint(2**31 - 1)))
        t = 0
        done = False
        trajectory = []
        while t < T and not done:
            a = policy.draw_action(s)
            s1, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            trajectory.append((s, a, r))
            s = s1
            t += 1
        trajectories.append(trajectory)
    return trajectories


def animate(data, interval=40):
    fig = plt.figure(1)
    img = plt.imshow(data[0])
    plt.axis('off')
    def _step(i):
        img.set_data(data[i])
    anim = animation.FuncAnimation(fig, _step, frames=len(data), interval=interval)
    plt.close(1)
    return anim
