# cliffwalking.py
import numpy as np
from enum import Enum
import random

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class CliffWalking:
    def __init__(self, width=12, height=4):
        self.width = width
        self.height = height
        self.size = max(width, height)  # Add for compatibility with RLAlgorithm
        self.n_states = width * height
        self.n_actions = 4
        # ... rest of the code stays the same ...
        
        # Start at bottom-left, goal at bottom-right
        self.start_state = [height - 1, 0]
        self.goal_state = [height - 1, width - 1]
        
        # Cliff: bottom row except start and goal
        self.cliff_cells = []
        for col in range(1, width - 1):
            self.cliff_cells.append([height - 1, col])
        
        # Rewards
        self.goal_reward = 10.0
        self.cliff_reward = -100.0
        self.step_reward = -1.0
        self.wall_reward = -1.0
        
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
        
        # Calculate intended movement
        if action == Action.UP.value:
            x = max(0, x - 1)
        elif action == Action.DOWN.value:
            x = min(self.height - 1, x + 1)
        elif action == Action.LEFT.value:
            y = max(0, y - 1)
        elif action == Action.RIGHT.value:
            y = min(self.width - 1, y + 1)
        
        next_state = [x, y]
        
        # Check for cliff
        if next_state in self.cliff_cells:
            reward = self.cliff_reward
            self.done = True
            # Move agent back to start after falling off cliff
            next_state = self.start_state.copy()
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
    
    def get_available_actions(self, state):
        """Get all valid actions from a state"""
        x, y = state
        actions = []
        
        if x > 0:
            actions.append(Action.UP.value)
        if x < self.height - 1:
            actions.append(Action.DOWN.value)
        if y > 0:
            actions.append(Action.LEFT.value)
        if y < self.width - 1:
            actions.append(Action.RIGHT.value)
            
        return actions
    
    def simulate_step(self, state, action):
        """Simulate a step without modifying internal state"""
        x, y = state
        
        if action == Action.UP.value:
            x = max(0, x - 1)
        elif action == Action.DOWN.value:
            x = min(self.height - 1, x + 1)
        elif action == Action.LEFT.value:
            y = max(0, y - 1)
        elif action == Action.RIGHT.value:
            y = min(self.width - 1, y + 1)
        
        next_state = [x, y]
        
        # Check for cliff
        if next_state in self.cliff_cells:
            reward = self.cliff_reward
            done = True
            next_state = self.start_state.copy()
        elif next_state == self.goal_state:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False
        
        return next_state, reward, done
    
    def get_grid_info(self):
        return {
            "width": self.width,
            "height": self.height,
            "start": self.start_state,
            "goal": self.goal_state,
            "cliff_cells": self.cliff_cells,
            "agent_position": self.state,
            "cliff_count": len(self.cliff_cells),
            "goal_reward": self.goal_reward,
            "cliff_reward": self.cliff_reward,
            "step_reward": self.step_reward
        }
    
    def render(self):
        """Render the current grid state"""
        grid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                pos = [i, j]
                if pos == self.state:
                    row.append('A')
                elif pos == self.start_state:
                    row.append('S')
                elif pos == self.goal_state:
                    row.append('G')
                elif pos in self.cliff_cells:
                    row.append('C')
                else:
                    row.append('.')
            grid.append(row)
        return grid