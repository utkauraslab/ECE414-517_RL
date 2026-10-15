import numpy as np
from gymnasium import Env

UP    = 0
RIGHT = 1
DOWN  = 2
LEFT  = 3

class CliffWalkingEnv(Env):
    ROWS = 4
    COLS = 12
    CLIFF = {(3, c) for c in range(1, 11)}
    GOAL = (3, 11)

    def reset(self, *, seed=None, options=None):
        valid_states = [
            (r, c)
            for r in range(self.ROWS)
            for c in range(self.COLS)
            if (r, c) not in self.CLIFF and (r, c) != self.GOAL
        ]
        self.state = valid_states[np.random.randint(len(valid_states))]
        self.interactions = 0
        return self.state, {}

    def step(self, action: int):
        r, c = self.state
        if   action == UP    and r > 0:            self.state = (r - 1, c)
        elif action == RIGHT and c < self.COLS - 1: self.state = (r, c + 1)
        elif action == DOWN  and r < self.ROWS - 1: self.state = (r + 1, c)
        elif action == LEFT  and c > 0:             self.state = (r, c - 1)

        self.interactions += 1

        terminated = self.state == self.GOAL
        truncated  = self.interactions >= 50
        reward     = -100 if self.state in self.CLIFF else -1

        if reward == -100:
            terminated = True

        return self.state, reward, terminated, truncated, {}
