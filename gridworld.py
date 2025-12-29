# gridworld.py - FINAL FIXED VERSION (COMPATIBLE)

import random
from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld:
    def __init__(self, size=5, random_walls=True, wall_density=0.1):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4

        self.start_state = [0, 0]
        self.goal_state = [size - 1, size - 1]

        # Walls
        self.walls = []
        if random_walls and size >= 5:
            num_walls = int(size * size * wall_density)
            attempts = 0
            while len(self.walls) < num_walls and attempts < 200:
                wall = [random.randint(0, size - 1), random.randint(0, size - 1)]
                if (
                    wall != self.start_state
                    and wall != self.goal_state
                    and wall not in self.walls
                ):
                    self.walls.append(wall)
                attempts += 1

        # Ensure diagonal path exists
        for i in range(size):
            if [i, i] in self.walls:
                self.walls.remove([i, i])

        # Rewards
        self.goal_reward = 10.0
        self.step_reward = -0.1
        self.wall_reward = -1.0

        self.reset()

    def reset(self):
        self.state = self.start_state.copy()
        self.total_reward = 0.0
        self.steps = 0
        return self.state

    def step(self, action: int):
        if self.state == self.goal_state:
            return self.state, 0.0, True

        x, y = self.state
        prev_state = self.state.copy()

        if action == Action.UP.value:
            x = max(0, x - 1)
        elif action == Action.DOWN.value:
            x = min(self.size - 1, x + 1)
        elif action == Action.LEFT.value:
            y = max(0, y - 1)
        elif action == Action.RIGHT.value:
            y = min(self.size - 1, y + 1)

        next_state = [x, y]

        if next_state in self.walls:
            reward = self.wall_reward
            next_state = prev_state
            done = False
        elif next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        self.state = next_state
        self.total_reward += reward
        self.steps += 1

        return self.state, reward, done

    def get_available_actions(self, state):
        return list(range(4))

    def get_grid_info(self):
        return {
            "size": self.size,
            "start": self.start_state,
            "goal": self.goal_state,
            "walls": self.walls,
            "agent_position": self.state,
            "wall_count": len(self.walls),
        }
    
    def simulate_step(self, state, action):
        """Simulate a step without modifying internal state"""
        x, y = state
        
        if action == Action.UP.value:
            x = max(0, x - 1)
        elif action == Action.DOWN.value:
            x = min(self.size - 1, x + 1)
        elif action == Action.LEFT.value:
            y = max(0, y - 1)
        elif action == Action.RIGHT.value:
            y = min(self.size - 1, y + 1)
        
        next_state = [x, y]
        
        # Check for walls
        if next_state in self.walls:
            reward = self.wall_reward
            next_state = state  # Stay in same position
            done = False
        elif next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False
            
        return next_state, reward, done
    
    def get_available_actions(self, state):
        """Get all valid actions from a state"""
        x, y = state
        actions = []
        
        # Check UP
        if x > 0 and [x-1, y] not in self.walls:
            actions.append(Action.UP.value)
        # Check DOWN
        if x < self.size-1 and [x+1, y] not in self.walls:
            actions.append(Action.DOWN.value)
        # Check LEFT
        if y > 0 and [x, y-1] not in self.walls:
            actions.append(Action.LEFT.value)
        # Check RIGHT
        if y < self.size-1 and [x, y+1] not in self.walls:
            actions.append(Action.RIGHT.value)
            
        return actions
