# frozenlake.py - FIXED VERSION
import numpy as np
from enum import Enum
import random

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class FrozenLake:
    def __init__(self, size=4, is_slippery=True, hole_prob=0.2):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.is_slippery = is_slippery
        
        # Start at top-left, goal at bottom-right
        self.start_state = [0, 0]
        self.goal_state = [size - 1, size - 1]
        
        # Generate holes
        self.holes = []
        if hole_prob > 0:
            num_holes = int(size * size * hole_prob)
            attempts = 0
            while len(self.holes) < num_holes and attempts < 200:
                hole = [random.randint(0, size - 1), random.randint(0, size - 1)]
                if (hole != self.start_state and 
                    hole != self.goal_state and 
                    hole not in self.holes):
                    # Ensure there's a path from start to goal
                    # Simple check: don't block entire first row/column
                    if not (hole[0] == 0 and hole[1] == 1) and not (hole[0] == 1 and hole[1] == 0):
                        self.holes.append(hole)
                attempts += 1
        
        # Rewards
        self.goal_reward = 10.0
        self.hole_reward = -10.0
        self.step_reward = -0.1
        
        self.reset()
    
    def reset(self):
        self.state = self.start_state.copy()
        self.total_reward = 0.0
        self.steps = 0
        self.done = False
        return self.state
    
    def step(self, action: int):
        if self.done:
            return self.state, 0.0, True
        
        x, y = self.state
        
        # Apply slippery condition ONLY if slippery is enabled
        if self.is_slippery and random.random() < 0.33:
            # 33% chance to slip to a random direction
            possible_actions = [0, 1, 2, 3]
            possible_actions.remove(action)
            action = random.choice(possible_actions)
        
        # Calculate movement
        if action == Action.UP.value:
            x = max(0, x - 1)
        elif action == Action.DOWN.value:
            x = min(self.size - 1, x + 1)
        elif action == Action.LEFT.value:
            y = max(0, y - 1)
        elif action == Action.RIGHT.value:
            y = min(self.size - 1, y + 1)
        
        next_state = [x, y]
        
        # Check for holes
        if next_state in self.holes:
            reward = self.hole_reward
            self.done = True
        elif next_state == self.goal_state:
            reward = self.goal_reward
            self.done = True
        else:
            reward = self.step_reward
            self.done = False
        
        self.state = next_state
        self.total_reward += reward
        self.steps += 1
        
        return self.state, reward, self.done
    
    def simulate_step(self, state, action):
        """Simulate a step without modifying internal state - FIXED for deterministic"""
        x, y = state
        
        # For deterministic simulation (used by DP algorithms), don't apply slippery
        # OR apply it deterministically for planning purposes
        # For FrozenLake, when slippery=False, it should be deterministic
        
        if action == Action.UP.value:
            x = max(0, x - 1)
        elif action == Action.DOWN.value:
            x = min(self.size - 1, x + 1)
        elif action == Action.LEFT.value:
            y = max(0, y - 1)
        elif action == Action.RIGHT.value:
            y = min(self.size - 1, y + 1)
        
        next_state = [x, y]
        
        # Check for holes
        if next_state in self.holes:
            reward = self.hole_reward
            done = True
        elif next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False
        
        return next_state, reward, done
    
    def get_available_actions(self, state):
        """Get all valid actions from a state"""
        return list(range(4))  # All actions available in FrozenLake
    
    def get_grid_info(self):
        return {
            "size": self.size,
            "start": self.start_state,
            "goal": self.goal_state,
            "holes": self.holes,
            "agent_position": self.state,
            "hole_count": len(self.holes),
            "is_slippery": self.is_slippery,
            "goal_reward": self.goal_reward,
            "hole_reward": self.hole_reward,
            "step_reward": self.step_reward
        }
    
    def render(self):
        """Render the current grid state"""
        grid = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                pos = [i, j]
                if pos == self.state:
                    row.append('A')
                elif pos == self.start_state:
                    row.append('S')
                elif pos == self.goal_state:
                    row.append('G')
                elif pos in self.holes:
                    row.append('H')
                else:
                    row.append('F')
            grid.append(row)
        return grid