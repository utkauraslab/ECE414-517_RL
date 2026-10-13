from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from abc import ABC, abstractmethod
import seaborn as sns
from PIL import Image
import urllib.request
from io import BytesIO
from IPython.display import clear_output
from gymnasium import spaces

# We import the rendering logic and constants from our hidden black box
from rl_envs.taxi_utils import render_taxi, close_taxi_render

UP    = 0
RIGHT = 1
DOWN  = 2
LEFT  = 3

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def setup():
    register(
        id="CliffWalking-RLSS-v0",
        entry_point="rl_envs.custom_cliff:CliffWalkingEnv",
    )

def animate_frames(frames, interval=200):
    """
    Anima una lista di frame RGB ottenuti da un env Gymnasium.

    Args:
        frames:   lista di array numpy (H, W, 3) ottenuti con env.render()
        interval: ms tra un frame e l'altro
    """
    h, w = frames[0].shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_plot = ax.imshow(frames[0])

    def update(frame_idx):
        img_plot.set_data(frames[frame_idx])
        return [img_plot]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        repeat=True
    )

    plt.close(fig)
    display(HTML('<center>' + ani.to_jshtml(default_mode='loop') + '</center>'))

def plot_cumulative_reward(mean_cumulative_rewards: list, margin_of_error: list):
    # Convert to numpy arrays for element-wise math operations
    mean_cumulative_rewards = np.array(mean_cumulative_rewards)
    std_cumulative_rewards = np.array(std_cumulative_rewards)

    plt.figure(figsize=(10, 6))
    episodes = range(len(mean_cumulative_rewards))
    
    # Plot the mean line
    plt.plot(episodes, mean_cumulative_rewards, label='Mean Cumulative Reward', color='blue')
    
    # Plot the 95% Confidence Interval shaded region
    plt.fill_between(
        episodes,
        mean_cumulative_rewards - margin_of_error,
        mean_cumulative_rewards + margin_of_error,
        alpha=0.3,
        color='blue',
        label='95% Confidence Interval'
    )
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Agent Performance Over Time (with 95% CI)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_gridworld_values(V, gif_path=None, title="Value Function Estimation", cbar_label="Estimated Value V(s)"):
    """
    Plots the estimated Value function for a Gridworld where V is a 2D numpy array.
    Overlays the heatmap on a static frame from a given GIF, enforcing the image's native aspect ratio.
    """
    num_rows, num_cols = V.shape
    
    annot_grid = np.empty((num_rows, num_cols), dtype=object)
    for r in range(num_rows):
        for c in range(num_cols):
            val = V[r, c]
            if np.isnan(val) or np.isinf(val):
                annot_grid[r, c] = ""
            else:
                annot_grid[r, c] = f"{val:.1f}"
                
    # --- 1. LOAD BACKGROUND FIRST TO EXTRACT DIMENSIONS ---
    img = None
    heatmap_alpha = 1.0
    
    if gif_path:
        try:
            if gif_path.startswith('http://') or gif_path.startswith('https://'):
                req = urllib.request.Request(gif_path, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    img_data = response.read()
                img = Image.open(BytesIO(img_data)).convert("RGB")
            else:
                img = Image.open(gif_path).convert("RGB")
            heatmap_alpha = 0.7
        except Exception as e:
            print(f"Warning: Could not load background image from {gif_path}. Error: {e}")

    # --- 2. CALCULATE DYNAMIC SIZE & TRUE ASPECT RATIO ---
    target_aspect = 'auto'
    
    if img:
        img_width, img_height = img.size
        
        # Calculate the exact aspect parameter to preserve the image's native proportions
        # Matplotlib aspect = (physical_height / physical_width) * (data_width / data_height)
        target_aspect = (img_height / img_width) * (num_cols / num_rows)
        
        # Adjust figure size to roughly match the image aspect (adding 20% width for the colorbar)
        base_width = max(8, num_cols * 1.0)
        fig_height = base_width * (img_height / img_width)
        fig_width = base_width * 1.2 
        plt.figure(figsize=(fig_width, fig_height))
    else:
        plt.figure(figsize=(max(8, num_cols * 1.2), max(6, num_rows * 1.2)))
        
    ax = plt.gca()
    
    # --- 3. PLOT BACKGROUND IMAGE WITH FIXED ASPECT ---
    if img:
        # Pass the calculated aspect ratio so the image doesn't distort
        ax.imshow(img, extent=[0, num_cols, num_rows, 0], aspect=target_aspect, zorder=0)

    # --- 4. PLOT HEATMAP ---
    cmap = "magma" 
    sns.heatmap(V, annot=annot_grid, fmt="", cmap=cmap, 
                cbar_kws={'label': cbar_label}, 
                linewidths=1, linecolor='gray',
                alpha=heatmap_alpha, zorder=1, ax=ax)
    
    # Enforce the aspect ratio on the axes just in case seaborn tries to reset it
    if img:
        ax.set_aspect(target_aspect)
        
    ax.set_title(title, fontsize=14, pad=15)
    
    # Hide the standard coordinate tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def plot_training_results(alg_name, mean_returns, confidence_intervals, eval_every=20):
    sns.set_theme(style="darkgrid")
    episodes = np.arange(1, len(mean_returns) + 1) * eval_every
    means = np.array(mean_returns)
    cis = np.array(confidence_intervals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, means, label="Average Cumulative Return", color="#2c7bb6", linewidth=2)
    
    # Updated label to reflect the 95% Confidence Interval
    plt.fill_between(episodes, means - cis, means + cis, color="#2c7bb6", alpha=0.2, label="95% Confidence Interval")
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Cumulative Return", fontsize=12)
    plt.title(f"{alg_name} Training Performance", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    
    plt.tight_layout()
    plt.show()