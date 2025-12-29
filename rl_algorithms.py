import numpy as np
import random

class RLAlgorithm:
    def __init__(self, env):
        self.env = env
        self.n_actions = env.n_actions
        
        # Handle different environment types
        if hasattr(env, 'size'):
            self.size = env.size
            self.grid_width = env.size
            self.grid_height = env.size
        elif hasattr(env, 'width') and hasattr(env, 'height'):
            self.grid_width = env.width
            self.grid_height = env.height
            self.size = max(env.width, env.height)
        else:
            raise ValueError("Environment must have either 'size' or 'width'/'height' attributes")
    
    def get_valid_actions(self, state):
        """Get valid actions from a state"""
        return self.env.get_available_actions(state)
    
    def q_learning(self, episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1, return_history=False):
        """Q-learning algorithm (off-policy TD control)"""
        print(f"Starting Q-learning with {episodes} episodes")
        
        Q = np.random.randn(self.grid_height, self.grid_width, self.n_actions) * 0.01
        history = {'episodes': [], 'rewards': [], 'max_q_values': [], 'steps': [], 'epsilon_values': []}
        
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            # Decay epsilon
            eps = max(0.01, epsilon * (1 - ep/episodes))
            
            while not done and steps < 500:
                x, y = state
                
                # Epsilon-greedy action selection
                if np.random.rand() < eps:
                    valid_actions = self.get_valid_actions(state)
                    action = np.random.choice(valid_actions) if valid_actions else 0
                else:
                    action = np.argmax(Q[x, y])
                
                # Take action
                next_state, reward, done = self.env.step(action)
                nx, ny = next_state
                
                # Q-learning update
                if done:
                    td_target = reward
                else:
                    td_target = reward + gamma * np.max(Q[nx, ny])
                
                # Update Q-value
                Q[x, y, action] += alpha * (td_target - Q[x, y, action])
                
                # Move to next state
                state = next_state
                total_reward += reward
                steps += 1
            
            # Record history
            if ep % 10 == 0 or ep == episodes-1:
                history['episodes'].append(ep)
                history['rewards'].append(total_reward)
                history['max_q_values'].append(np.max(Q))
                history['steps'].append(steps)
                history['epsilon_values'].append(eps)
            
            # Print progress
            if ep % 100 == 0:
                print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {eps:.3f}")
        
        # Extract policy and value function
        policy = np.argmax(Q, axis=2)
        value_function = np.max(Q, axis=2)
        
        result = {
            'Q': Q.tolist(),
            'policy': policy.tolist(),
            'value_function': value_function.tolist(),
            'algorithm': 'q-learning',
            'parameters': {'episodes': episodes, 'gamma': gamma, 'alpha': alpha, 'epsilon': epsilon}
        }
        
        return (result, history) if return_history else result
    
    def sarsa(self, episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1, return_history=False):
        """SARSA algorithm (on-policy TD control)"""
        print(f"Starting SARSA with {episodes} episodes")
        
        Q = np.random.randn(self.grid_height, self.grid_width, self.n_actions) * 0.01
        history = {'episodes': [], 'rewards': [], 'max_q_values': [], 'steps': [], 'epsilon_values': []}
        
        for ep in range(episodes):
            state = self.env.reset()
            x, y = state
            total_reward = 0
            steps = 0
            done = False
            
            # Choose initial action with epsilon-greedy
            if np.random.rand() < epsilon:
                valid_actions = self.get_valid_actions(state)
                action = np.random.choice(valid_actions) if valid_actions else 0
            else:
                action = np.argmax(Q[x, y])
            
            # Decay epsilon
            eps = max(0.01, epsilon * (1 - ep/episodes))
            
            while not done and steps < 500:
                # Take action
                next_state, reward, done = self.env.step(action)
                nx, ny = next_state
                
                # Choose next action with epsilon-greedy
                if np.random.rand() < eps:
                    valid_actions = self.get_valid_actions(next_state)
                    next_action = np.random.choice(valid_actions) if valid_actions else 0
                else:
                    next_action = np.argmax(Q[nx, ny])
                
                # SARSA update
                if done:
                    td_target = reward
                else:
                    td_target = reward + gamma * Q[nx, ny, next_action]
                
                Q[x, y, action] += alpha * (td_target - Q[x, y, action])
                
                # Update state and action
                state = next_state
                x, y = state
                action = next_action
                
                total_reward += reward
                steps += 1
            
            # Record history
            if ep % 10 == 0 or ep == episodes-1:
                history['episodes'].append(ep)
                history['rewards'].append(total_reward)
                history['max_q_values'].append(np.max(Q))
                history['steps'].append(steps)
                history['epsilon_values'].append(eps)
            
            if ep % 100 == 0:
                print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {eps:.3f}")
        
        policy = np.argmax(Q, axis=2)
        value_function = np.max(Q, axis=2)
        
        result = {
            'Q': Q.tolist(),
            'policy': policy.tolist(),
            'value_function': value_function.tolist(),
            'algorithm': 'sarsa',
            'parameters': {'episodes': episodes, 'gamma': gamma, 'alpha': alpha, 'epsilon': epsilon}
        }
        
        return (result, history) if return_history else result
    
    def value_iteration(self, gamma=0.99, theta=1e-6, max_iterations=1000, return_history=False):
        """Value Iteration algorithm (dynamic programming) - FIXED for stochastic"""
        print(f"Starting Value Iteration with gamma={gamma}")
        
        V = np.zeros((self.grid_height, self.grid_width))
        history = {'iterations': [], 'delta': [], 'max_values': []}
        iteration = 0
        
        while iteration < max_iterations:
            delta = 0
            
            # Iterate over all states
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    state = [i, j]
                    
                    # Skip terminal states
                    if self._is_terminal(state):
                        continue
                    
                    old_value = V[i, j]
                    
                    # Calculate value for each action
                    action_values = []
                    valid_actions = self.get_valid_actions(state)
                    
                    for action in valid_actions:
                        # Handle stochastic transitions if environment supports it
                        if hasattr(self.env, 'is_slippery') and self.env.is_slippery:
                            # For FrozenLake with slippery surface
                            # We need to consider all possible outcomes
                            action_value = 0.0
                            
                            # Intended direction (67% chance in FrozenLake)
                            intended_state, intended_reward, _ = self.env.simulate_step(state, action)
                            ni, nj = intended_state
                            action_value += 0.67 * (intended_reward + gamma * V[ni, nj])
                            
                            # Slipped directions (11% chance each for the other 3 directions)
                            other_actions = [a for a in valid_actions if a != action]
                            for other_action in other_actions:
                                slip_state, slip_reward, _ = self.env.simulate_step(state, other_action)
                                si, sj = slip_state
                                action_value += 0.11 * (slip_reward + gamma * V[si, sj])
                            
                            action_values.append(action_value)
                        else:
                            # Deterministic case
                            next_state, reward, _ = self.env.simulate_step(state, action)
                            nx, ny = next_state
                            action_value = reward + gamma * V[nx, ny]
                            action_values.append(action_value)
                    
                    # Update value with maximum action value
                    if action_values:
                        V[i, j] = max(action_values)
                    
                    delta = max(delta, abs(old_value - V[i, j]))
            
            iteration += 1
            history['iterations'].append(iteration)
            history['delta'].append(delta)
            history['max_values'].append(np.max(V))
            
            if delta < theta:
                print(f"Value iteration converged after {iteration} iterations with delta={delta}")
                break
        
        # Extract policy from value function
        policy = np.zeros((self.grid_height, self.grid_width), dtype=int)
        
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                state = [i, j]
                
                if self._is_terminal(state):
                    policy[i, j] = 0  # Default action
                    continue
                
                # Find best action
                best_action = 0
                best_value = -np.inf
                valid_actions = self.get_valid_actions(state)
                
                for action in valid_actions:
                    # Handle stochastic transitions
                    if hasattr(self.env, 'is_slippery') and self.env.is_slippery:
                        action_value = 0.0
                        intended_state, intended_reward, _ = self.env.simulate_step(state, action)
                        ni, nj = intended_state
                        action_value += 0.67 * (intended_reward + gamma * V[ni, nj])
                        
                        other_actions = [a for a in valid_actions if a != action]
                        for other_action in other_actions:
                            slip_state, slip_reward, _ = self.env.simulate_step(state, other_action)
                            si, sj = slip_state
                            action_value += 0.11 * (slip_reward + gamma * V[si, sj])
                    else:
                        next_state, reward, _ = self.env.simulate_step(state, action)
                        nx, ny = next_state
                        action_value = reward + gamma * V[nx, ny]
                    
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                
                policy[i, j] = best_action
        
        result = {
            'V': V.tolist(),
            'policy': policy.tolist(),
            'value_function': V.tolist(),
            'algorithm': 'value-iteration',
            'parameters': {'gamma': gamma, 'theta': theta, 'iterations': iteration}
        }
        
        return (result, history) if return_history else result
    
    def policy_iteration(self, gamma=0.99, theta=1e-6, max_iterations=1000, return_history=False):
        """Policy Iteration algorithm (dynamic programming) - FIXED for stochastic"""
        print(f"Starting Policy Iteration with gamma={gamma}")
        
        # Initialize random policy
        policy = np.random.randint(0, self.n_actions, (self.grid_height, self.grid_width))
        V = np.zeros((self.grid_height, self.grid_width))
        history = {'iterations': [], 'policy_changes': [], 'max_values': []}
        
        for iteration in range(max_iterations):
            # Policy Evaluation
            for _ in range(10):  # Limited evaluation iterations
                delta = 0
                
                for i in range(self.grid_height):
                    for j in range(self.grid_width):
                        state = [i, j]
                        
                        # Skip terminal states
                        if self._is_terminal(state):
                            if hasattr(self.env, 'goal_state') and state == self.env.goal_state:
                                V[i, j] = self.env.goal_reward
                            continue
                        
                        old_value = V[i, j]
                        action = policy[i, j]
                        
                        # Check if action is valid
                        if action not in self.get_valid_actions(state):
                            # Choose a valid action
                            valid_actions = self.get_valid_actions(state)
                            action = valid_actions[0] if valid_actions else 0
                        
                        # Evaluate current policy with stochastic transitions
                        if hasattr(self.env, 'is_slippery') and self.env.is_slippery:
                            expected_value = 0.0
                            intended_state, intended_reward, _ = self.env.simulate_step(state, action)
                            ni, nj = intended_state
                            expected_value += 0.67 * (intended_reward + gamma * V[ni, nj])
                            
                            other_actions = [a for a in self.get_valid_actions(state) if a != action]
                            for other_action in other_actions:
                                slip_state, slip_reward, _ = self.env.simulate_step(state, other_action)
                                si, sj = slip_state
                                expected_value += 0.11 * (slip_reward + gamma * V[si, sj])
                            
                            V[i, j] = expected_value
                        else:
                            # Deterministic evaluation
                            next_state, reward, _ = self.env.simulate_step(state, action)
                            nx, ny = next_state
                            V[i, j] = reward + gamma * V[nx, ny]
                        
                        delta = max(delta, abs(old_value - V[i, j]))
                
                if delta < theta:
                    break
            
            # Policy Improvement
            policy_changed = False
            
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    state = [i, j]
                    
                    if self._is_terminal(state):
                        continue
                    
                    old_action = policy[i, j]
                    best_action = old_action
                    best_value = -np.inf
                    valid_actions = self.get_valid_actions(state)
                    
                    for action in valid_actions:
                        # Calculate action value with stochastic transitions
                        if hasattr(self.env, 'is_slippery') and self.env.is_slippery:
                            action_value = 0.0
                            intended_state, intended_reward, _ = self.env.simulate_step(state, action)
                            ni, nj = intended_state
                            action_value += 0.67 * (intended_reward + gamma * V[ni, nj])
                            
                            other_actions = [a for a in valid_actions if a != action]
                            for other_action in other_actions:
                                slip_state, slip_reward, _ = self.env.simulate_step(state, other_action)
                                si, sj = slip_state
                                action_value += 0.11 * (slip_reward + gamma * V[si, sj])
                        else:
                            next_state, reward, _ = self.env.simulate_step(state, action)
                            nx, ny = next_state
                            action_value = reward + gamma * V[nx, ny]
                        
                        if action_value > best_value:
                            best_value = action_value
                            best_action = action
                    
                    policy[i, j] = best_action
                    if old_action != best_action:
                        policy_changed = True
            
            history['iterations'].append(iteration + 1)
            history['policy_changes'].append(policy_changed)
            history['max_values'].append(np.max(V))
            
            if not policy_changed:
                print(f"Policy iteration converged after {iteration + 1} iterations")
                break
        
        result = {
            'V': V.tolist(),
            'policy': policy.tolist(),
            'value_function': V.tolist(),
            'algorithm': 'policy-iteration',
            'parameters': {'gamma': gamma, 'theta': theta, 'iterations': iteration + 1}
        }
        
        return (result, history) if return_history else result
    
    def monte_carlo(self, episodes=2000, gamma=0.99, epsilon=0.1, return_history=False):
        """Monte Carlo First-Visit control algorithm - IMPROVED for FrozenLake"""
        print(f"Starting Monte Carlo First-Visit with {episodes} episodes")
        
        Q = np.random.randn(self.grid_height, self.grid_width, self.n_actions) * 0.01
        returns = {(i, j, a): [] for i in range(self.grid_height) for j in range(self.grid_width) for a in range(self.n_actions)}
        history = {'episodes': [], 'rewards': [], 'steps': [], 'epsilon_values': []}
        
        for ep in range(episodes):
            state = self.env.reset()
            episode = []
            total_reward = 0
            steps = 0
            done = False
            
            # Decay epsilon
            eps = max(0.01, epsilon * (1 - ep/episodes))
            
            # Generate episode
            while not done and steps < 500:
                x, y = state
                
                # Epsilon-greedy action selection with exploration bonus for early episodes
                if np.random.rand() < eps or ep < 100:  # More exploration early
                    valid_actions = self.get_valid_actions(state)
                    action = np.random.choice(valid_actions) if valid_actions else 0
                else:
                    action = np.argmax(Q[x, y])
                
                next_state, reward, done = self.env.step(action)
                
                episode.append((state, action, reward))
                state = next_state
                total_reward += reward
                steps += 1
            
            # Update Q-values using first-visit Monte Carlo
            G = 0
            visited_states_actions = set()
            
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                x, y = state
                G = gamma * G + reward
                
                # First-visit: only update first time state-action pair is visited in episode
                if (x, y, action) not in visited_states_actions:
                    visited_states_actions.add((x, y, action))
                    returns[(x, y, action)].append(G)
                    # Use incremental mean for stability
                    if len(returns[(x, y, action)]) == 1:
                        Q[x, y, action] = G
                    else:
                        Q[x, y, action] += (G - Q[x, y, action]) / len(returns[(x, y, action)])
            
            # Record history
            if ep % 10 == 0 or ep == episodes-1:
                history['episodes'].append(ep)
                history['rewards'].append(total_reward)
                history['steps'].append(steps)
                history['epsilon_values'].append(eps)
            
            if ep % 200 == 0:
                print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {eps:.3f}")
        
        # Extract policy and value function
        policy = np.argmax(Q, axis=2)
        value_function = np.max(Q, axis=2)
        
        result = {
            'Q': Q.tolist(),
            'policy': policy.tolist(),
            'value_function': value_function.tolist(),
            'algorithm': 'monte-carlo',
            'parameters': {'episodes': episodes, 'gamma': gamma, 'epsilon': epsilon}
        }
        
        return (result, history) if return_history else result
    
    def td_learning(self, episodes=2000, gamma=0.99, alpha=0.1, epsilon=0.1, return_history=False):
        """Temporal Difference (TD) learning (Same as SARSA for this implementation)"""
        print(f"Starting TD Learning with {episodes} episodes")
        
        # TD learning is essentially SARSA for this implementation
        return self.sarsa(episodes=episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, return_history=return_history)
    
    def nstep_td(self, episodes=2000, gamma=0.99, alpha=0.1, epsilon=0.1, n_steps=3, return_history=False):
        """N-step Temporal Difference learning"""
        print(f"Starting {n_steps}-step TD Learning with {episodes} episodes")
        
        Q = np.random.randn(self.grid_height, self.grid_width, self.n_actions) * 0.01
        history = {'episodes': [], 'rewards': [], 'steps': [], 'epsilon_values': []}
        
        for ep in range(episodes):
            state = self.env.reset()
            x, y = state
            total_reward = 0
            steps = 0
            done = False
            
            # Store n-step experience
            states = []
            actions = []
            rewards = []
            
            # Choose initial action
            if np.random.rand() < epsilon:
                valid_actions = self.get_valid_actions(state)
                action = np.random.choice(valid_actions) if valid_actions else 0
            else:
                action = np.argmax(Q[x, y])
            
            # Decay epsilon
            eps = max(0.01, epsilon * (1 - ep/episodes))
            
            T = float('inf')
            t = 0
            
            while not done and steps < 500:
                if t < T:
                    # Take action
                    next_state, reward, done = self.env.step(action)
                    nx, ny = next_state
                    
                    # Store experience
                    states.append([x, y])
                    actions.append(action)
                    rewards.append(reward)
                    
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        T = t + 1
                    else:
                        # Choose next action
                        if np.random.rand() < eps:
                            valid_actions = self.get_valid_actions(next_state)
                            next_action = np.random.choice(valid_actions) if valid_actions else 0
                        else:
                            next_action = np.argmax(Q[nx, ny])
                        
                        action = next_action
                        x, y = nx, ny
                
                tau = t - n_steps + 1
                
                if tau >= 0:
                    # Calculate n-step return
                    G = 0
                    for i in range(tau + 1, min(tau + n_steps, T) + 1):
                        G += (gamma ** (i - tau - 1)) * rewards[i-1]
                    
                    if tau + n_steps < T:
                        s_tau_n = states[tau + n_steps - 1]
                        a_tau_n = actions[tau + n_steps - 1]
                        G += (gamma ** n_steps) * Q[s_tau_n[0], s_tau_n[1], a_tau_n]
                    
                    # Update Q-value
                    s_tau = states[tau]
                    a_tau = actions[tau]
                    Q[s_tau[0], s_tau[1], a_tau] += alpha * (G - Q[s_tau[0], s_tau[1], a_tau])
                
                t += 1
                if tau == T - 1:
                    break
            
            # Record history
            if ep % 10 == 0 or ep == episodes-1:
                history['episodes'].append(ep)
                history['rewards'].append(total_reward)
                history['steps'].append(steps)
                history['epsilon_values'].append(eps)
            
            if ep % 200 == 0:
                print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {eps:.3f}")
        
        policy = np.argmax(Q, axis=2)
        value_function = np.max(Q, axis=2)
        
        result = {
            'Q': Q.tolist(),
            'policy': policy.tolist(),
            'value_function': value_function.tolist(),
            'algorithm': 'nstep-td',
            'parameters': {'episodes': episodes, 'gamma': gamma, 'alpha': alpha, 'epsilon': epsilon, 'n_steps': n_steps}
        }
        
        return (result, history) if return_history else result
    
    def _is_terminal(self, state):
        """Check if a state is terminal"""
        if hasattr(self.env, 'goal_state') and state == self.env.goal_state:
            return True
        if hasattr(self.env, 'holes') and state in self.env.holes:
            return True
        if hasattr(self.env, 'walls') and state in self.env.walls:
            return True
        if hasattr(self.env, 'cliff_cells') and state in self.env.cliff_cells:
            return True
        return False
    
    def get_algorithm_method(self, algorithm_name):
        """Get the algorithm method by name"""
        algorithm_map = {
            'q-learning': self.q_learning,
            'sarsa': self.sarsa,
            'value-iteration': self.value_iteration,
            'policy-iteration': self.policy_iteration,
            'monte-carlo': self.monte_carlo,
            'td-learning': self.td_learning,
            'nstep-td': self.nstep_td
        }
        
        # Convert kebab-case to snake_case
        snake_case_name = algorithm_name.replace('-', '_')
        
        # Try both naming conventions
        if algorithm_name in algorithm_map:
            return algorithm_map[algorithm_name]
        elif hasattr(self, snake_case_name):
            return getattr(self, snake_case_name)
        else:
            return None