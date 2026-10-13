from contextlib import closing
from io import StringIO
from os import path
from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces
from gymnasium.error import DependencyNotInstalled

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

POSITION_MAPPING = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

class CliffWalkingEnv(Env):
    """
    Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: str | None = None, is_slippery: bool = False):
        self.shape = (4, 12)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        self.is_slippery = is_slippery

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, UP)
            self.P[s][RIGHT] = self._calculate_transition_prob(position, RIGHT)
            self.P[s][DOWN] = self._calculate_transition_prob(position, DOWN)
            self.P[s][LEFT] = self._calculate_transition_prob(position, LEFT)

        self.initial_state_distrib = np.ones(self.nS) / 37
        for i in range(1, 12):
            self.initial_state_distrib[np.ravel_multi_index((3, i), self.shape)] = 0.

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.shape[0]),
            spaces.Discrete(self.shape[1]),
        ))

        self.render_mode = render_mode

        # pygame utils
        self.cell_size = (60, 60)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    def _limit_coordinates(self, coord: tuple[int, int]) -> tuple[int, int]:
        """Prevent the agent from falling out of the grid world."""
        row = min(max(coord[0], 0), self.shape[0] - 1)
        col = min(max(coord[1], 0), self.shape[1] - 1)
        return (row, col)

    def _state_to_xy(self, s: int) -> tuple[int, int]:
        row, col = np.unravel_index(s, self.shape)
        return (int(row), int(col))

    def _calculate_transition_prob(
        self, current: tuple[int, int] | np.ndarray, move: int
    ) -> list[tuple[float, Any, int, bool]]:
        if not self.is_slippery:
            deltas = [POSITION_MAPPING[move]]
        else:
            deltas = [
                POSITION_MAPPING[act] for act in [(move - 1) % 4, move, (move + 1) % 4]
            ]
        outcomes = []
        for delta in deltas:
            new_position = (int(current[0]) + delta[0], int(current[1]) + delta[1])
            new_position = self._limit_coordinates(new_position)
            new_state = np.ravel_multi_index(new_position, self.shape)
            if self._cliff[new_position]:
                outcomes.append((1 / len(deltas), self.start_state_index, -100, False))
            else:
                terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
                is_terminated = new_position == terminal_state
                outcomes.append((1 / len(deltas), new_state, -1, is_terminated))
        return outcomes

    def step(self, a):
        transitions = self.P[self.s][a]
        
        # Replaced categorical_sample with global np.random.choice
        probs = [t[0] for t in transitions]
        
        # Normalize probabilities in case of minor floating point rounding issues
        probs = np.array(probs) / np.sum(probs) 
        
        i = np.random.choice(len(transitions), p=probs)
        p, s, r, t = transitions[i]
        
        self.s = s
        self.lastaction = a
        self.counter += 1
        truncated = True if self.counter >= 50 else False

        if r == -100:
            t = True

        if self.render_mode == "human":
            self.render()
        
        return self._state_to_xy(int(s)), r, t, truncated, {"prob": p}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # Sync the global numpy seed if one is passed explicitly into reset()
        if seed is not None:
            np.random.seed(seed)

        # Replaced categorical_sample with global np.random.choice
        norm_distrib = np.array(self.initial_state_distrib) / np.sum(self.initial_state_distrib)
        self.s = np.random.choice(len(self.initial_state_distrib), p=norm_distrib)

        self.start_state_index = self.s
        self.lastaction = None
        self.counter = 0

        if self.render_mode == "human":
            self.render()
            
        return self._state_to_xy(int(self.s)), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "../imgs/custom_cliff/elf_up.png"),
                path.join(path.dirname(__file__), "../imgs/custom_cliff/elf_right.png"),
                path.join(path.dirname(__file__), "../imgs/custom_cliff/elf_down.png"),
                path.join(path.dirname(__file__), "../imgs/custom_cliff/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "../imgs/custom_cliff/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "../imgs/custom_cliff/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "../imgs/custom_cliff/mountain_bg1.png"),
                path.join(path.dirname(__file__), "../imgs/custom_cliff/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "../imgs/custom_cliff/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "../imgs/custom_cliff/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "../imgs/custom_cliff/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if row < self.shape[0] - 1 and self._cliff[row + 1, col]:
                self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def _render_text(self):
        outfile = StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
