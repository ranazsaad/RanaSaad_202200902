from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
import numpy as np
import threading
import time
from datetime import datetime
import sys
from cliffwalking import CliffWalking
import os
from frozenlake import FrozenLake
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Current directory:", os.path.dirname(os.path.abspath(__file__)))
print("Files in directory:", os.listdir(os.path.dirname(os.path.abspath(__file__))))

# Try to import RL algorithms - handle import errors gracefully
try:
    # First try to import from local files
    from gridworld import GridWorld
    print("Successfully imported GridWorld")
except ImportError as e:
    print(f"Error importing GridWorld: {e}")
    # Create a minimal GridWorld class
    class GridWorld:
        def __init__(self, size=5, random_walls=True, wall_density=0.1):
            self.size = size
            self.start_state = [0, 0]
            self.goal_state = [size-1, size-1]
            self.walls = []
            self.goal_reward = 10
            self.step_reward = -0.1
            self.wall_reward = -1
            self.n_actions = 4
            
            # Add some walls if random_walls is True
            if random_walls and size >= 5:
                import random
                num_walls = int(size * size * wall_density)
                for _ in range(num_walls):
                    wall = [random.randint(0, size-1), random.randint(0, size-1)]
                    if wall != self.start_state and wall != self.goal_state and wall not in self.walls:
                        self.walls.append(wall)

        def reset(self):
            return self.start_state.copy()

        def step(self, action):
            x, y = self.start_state
            if action == 0:
                x = max(0, x-1)
            elif action == 1:
                x = min(self.size-1, x+1)
            elif action == 2:
                y = max(0, y-1)
            elif action == 3:
                y = min(self.size-1, y+1)
            new_state = [x, y]
            
            if new_state in self.walls:
                reward = self.wall_reward
                new_state = self.start_state
                done = False
            elif new_state == self.goal_state:
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_reward
                done = False
                
            return new_state, reward, done

        def simulate_step(self, state, action):
            x, y = state
            if action == 0:
                x = max(0, x-1)
            elif action == 1:
                x = min(self.size-1, x+1)
            elif action == 2:
                y = max(0, y-1)
            elif action == 3:
                y = min(self.size-1, y+1)
            new_state = [x, y]
            
            if new_state in self.walls:
                reward = self.wall_reward
                new_state = state
                done = False
            elif new_state == self.goal_state:
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_reward
                done = False
                
            return new_state, reward, done

        def get_available_actions(self, state):
            x, y = state
            actions = []
            if x > 0: actions.append(0)  # Up
            if x < self.size-1: actions.append(1)  # Down
            if y > 0: actions.append(2)  # Left
            if y < self.size-1: actions.append(3)  # Right
            return actions if actions else [0, 1, 2, 3]

# Now import RLAlgorithm after GridWorld is defined
try:
    from rl_algorithms import RLAlgorithm
    print("Successfully imported RLAlgorithm")
except ImportError as e:
    print(f"Error importing RLAlgorithm: {e}")
    print("Creating fallback RLAlgorithm class...")
    
    # Create a complete RLAlgorithm class with all methods
    class RLAlgorithm:
        def __init__(self, env):
            self.env = env
            self.n_actions = env.n_actions
            self.size = env.size
            
        def simulate_step(self, state, action):
            x, y = state
            if action == 0: nx, ny = max(0, x-1), y
            elif action == 1: nx, ny = min(self.size-1, x+1), y
            elif action == 2: nx, ny = x, max(0, y-1)
            elif action == 3: nx, ny = x, min(self.size-1, y+1)
            
            new_state = [nx, ny]
            if new_state in self.env.walls:
                new_state = state
                reward = self.env.wall_reward
                done = False
            elif new_state == self.env.goal_state:
                reward = self.env.goal_reward
                done = True
            else:
                reward = self.env.step_reward
                done = False
                
            return new_state, reward, done
        
        def q_learning(self, episodes=1000, gamma=0.99, alpha=0.1, epsilon=0.1, return_history=False):
            print(f"Starting Q-learning with {episodes} episodes")
            Q = np.random.randn(self.size, self.size, self.n_actions) * 0.01
            history = {'episodes': [], 'rewards': [], 'max_q_values': [], 'steps': []}
            
            for ep in range(episodes):
                state = self.env.reset()
                total_reward = 0
                steps = 0
                done = False
                
                eps = epsilon * (1 - ep/episodes)
                
                while not done and steps < 200:
                    x, y = state
                    
                    # Epsilon-greedy
                    if np.random.rand() < eps:
                        action = np.random.randint(0, self.n_actions)
                    else:
                        action = np.argmax(Q[x, y])
                    
                    next_state, reward, done = self.env.step(action)
                    nx, ny = next_state
                    
                    # Q-learning update
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + gamma * np.max(Q[nx, ny])
                    
                    Q[x, y, action] += alpha * (td_target - Q[x, y, action])
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                if ep % 10 == 0 or ep == episodes-1:
                    history['episodes'].append(ep)
                    history['rewards'].append(total_reward)
                    history['max_q_values'].append(np.max(Q))
                    history['steps'].append(steps)
                    
                    if ep % 100 == 0:
                        print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {steps}")
            
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
            print(f"Starting SARSA with {episodes} episodes")
            Q = np.random.randn(self.size, self.size, self.n_actions) * 0.01
            history = {'episodes': [], 'rewards': [], 'max_q_values': [], 'steps': []}
            
            for ep in range(episodes):
                state = self.env.reset()
                x, y = state
                total_reward = 0
                steps = 0
                done = False
                
                # Choose initial action
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = np.argmax(Q[x, y])
                
                eps = epsilon * (1 - ep/episodes)
                
                while not done and steps < 200:
                    next_state, reward, done = self.env.step(action)
                    nx, ny = next_state
                    
                    # Choose next action
                    if np.random.rand() < eps:
                        next_action = np.random.randint(0, self.n_actions)
                    else:
                        next_action = np.argmax(Q[nx, ny])
                    
                    # SARSA update
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + gamma * Q[nx, ny, next_action]
                    
                    Q[x, y, action] += alpha * (td_target - Q[x, y, action])
                    
                    state, action = next_state, next_action
                    x, y = state
                    total_reward += reward
                    steps += 1
                
                if ep % 10 == 0 or ep == episodes-1:
                    history['episodes'].append(ep)
                    history['rewards'].append(total_reward)
                    history['max_q_values'].append(np.max(Q))
                    history['steps'].append(steps)
            
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
            print(f"Starting Value Iteration with gamma={gamma}")
            V = np.zeros((self.size, self.size))
            history = {'iterations': [], 'delta': []}
            iteration = 0
            
            while True:
                delta = 0
                for i in range(self.size):
                    for j in range(self.size):
                        state = [i, j]
                        
                        if state == self.env.goal_state:
                            V[i, j] = self.env.goal_reward
                            continue
                        if state in self.env.walls:
                            continue
                        
                        old_value = V[i, j]
                        
                        action_values = []
                        for a in range(self.n_actions):
                            next_state, reward, _ = self.simulate_step(state, a)
                            nx, ny = next_state
                            action_value = reward + gamma * V[nx, ny]
                            action_values.append(action_value)
                        
                        V[i, j] = max(action_values)
                        delta = max(delta, abs(old_value - V[i, j]))
                
                iteration += 1
                history['iterations'].append(iteration)
                history['delta'].append(delta)
                
                if delta < theta or iteration >= max_iterations:
                    print(f"Value iteration converged after {iteration} iterations")
                    break
            
            policy = np.zeros((self.size, self.size), dtype=int)
            for i in range(self.size):
                for j in range(self.size):
                    state = [i, j]
                    
                    if state == self.env.goal_state or state in self.env.walls:
                        policy[i, j] = 0
                        continue
                    
                    best_action = 0
                    best_value = -np.inf
                    
                    for a in range(self.n_actions):
                        next_state, reward, _ = self.simulate_step(state, a)
                        nx, ny = next_state
                        action_value = reward + gamma * V[nx, ny]
                        
                        if action_value > best_value:
                            best_value = action_value
                            best_action = a
                    
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
            print(f"Starting Policy Iteration with gamma={gamma}")
            policy = np.random.randint(0, self.n_actions, (self.size, self.size))
            V = np.zeros((self.size, self.size))
            history = {'iterations': [], 'policy_changes': []}
            
            for iteration in range(max_iterations):
                # Policy Evaluation
                while True:
                    delta = 0
                    for i in range(self.size):
                        for j in range(self.size):
                            state = [i, j]
                            
                            if state == self.env.goal_state:
                                V[i, j] = self.env.goal_reward
                                continue
                            if state in self.env.walls:
                                continue
                            
                            old_value = V[i, j]
                            action = policy[i, j]
                            
                            next_state, reward, _ = self.simulate_step(state, action)
                            nx, ny = next_state
                            
                            V[i, j] = reward + gamma * V[nx, ny]
                            delta = max(delta, abs(old_value - V[i, j]))
                    
                    if delta < theta:
                        break
                
                # Policy Improvement
                policy_changed = False
                for i in range(self.size):
                    for j in range(self.size):
                        state = [i, j]
                        
                        if state == self.env.goal_state or state in self.env.walls:
                            continue
                        
                        old_action = policy[i, j]
                        best_action = old_action
                        best_value = -np.inf
                        
                        for a in range(self.n_actions):
                            next_state, reward, _ = self.simulate_step(state, a)
                            nx, ny = next_state
                            action_value = reward + gamma * V[nx, ny]
                            
                            if action_value > best_value:
                                best_value = action_value
                                best_action = a
                        
                        policy[i, j] = best_action
                        if old_action != best_action:
                            policy_changed = True
                
                history['iterations'].append(iteration + 1)
                history['policy_changes'].append(policy_changed)
                
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

app = Flask(__name__)
app.secret_key = 'rl_tool_secret_key_2024'
CORS(app)

# Global training state
current_algorithm = None
training_active = False
training_results = {}
training_progress = 0
training_episode = 0
training_thread = None
training_history = []
training_environment = None  # Store environment globally instead of in session

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gridworld')
def gridworld():
    return render_template('gridworld.html')

@app.route('/api/select_algorithm', methods=['POST'])
def select_algorithm():
    global current_algorithm
    data = request.json
    algorithm = data.get('algorithm')
    params = data.get('parameters', {})
    
    current_algorithm = {
        'name': algorithm,
        'parameters': params,
        'status': 'selected',
        'selected_at': datetime.now().isoformat()
    }
    return jsonify({'status': 'success', 'algorithm': current_algorithm})

@app.route('/api/get_current_algorithm')
def get_current_algorithm():
    global current_algorithm
    if current_algorithm:
        return jsonify(current_algorithm)
    return jsonify({'name': 'None', 'status': 'not_selected'})

@app.route('/api/train', methods=['POST'])
def train():
    global training_active, current_algorithm, training_progress, training_episode, training_results, training_thread, training_history, training_environment
    
    if training_active:
        return jsonify({'error': 'Training already in progress'}), 400

    data = request.json
    algorithm_name = data.get('algorithm')
    params = data.get('parameters', {})

    size = 5
    
    try:
        # Create environment ONCE and reuse it
        if training_environment is None:
            print(f"Creating GridWorld environment with size={size}")
            training_environment = GridWorld(size=size, random_walls=True, wall_density=0.15)
        else:
            print(f"Reusing existing GridWorld environment")
        
        # Create RL algorithm instance with the SAME environment
        print(f"Creating RLAlgorithm instance for {algorithm_name}")
        rl = RLAlgorithm(training_environment)
        
        # Update current algorithm info
        current_algorithm = {
            'name': algorithm_name,
            'parameters': params,
            'status': 'training',
            'started_at': datetime.now().isoformat(),
            'progress': 0,
            'env_size': size
        }
        
        training_progress = 0
        training_episode = 0
        training_active = True
        training_history = []
        
        # Store environment info in session (NOT the object itself)
        session['gridworld_env'] = {
            'size': training_environment.size,
            'start': training_environment.start_state,
            'goal': training_environment.goal_state,
            'walls': training_environment.walls,
            'agent_position': training_environment.start_state.copy(),
            'goal_reward': training_environment.goal_reward,
            'step_reward': training_environment.step_reward,
            'wall_reward': training_environment.wall_reward
        }
        
        # Start training in background thread
        session_id = f"session_{int(time.time())}"
        print(f"Starting training thread for {algorithm_name} with session_id: {session_id}")
        
        training_thread = threading.Thread(
            target=run_training_thread, 
            args=(session_id, rl, algorithm_name, params)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'status': 'started', 
            'algorithm': algorithm_name, 
            'session_id': session_id,
            'message': f'Training {algorithm_name} started'
        })
        
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        traceback.print_exc()
        training_active = False
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

def run_training_thread(session_id, rl, algorithm_name, params):
    """Background thread for training"""
    global training_active, training_results, training_progress, training_episode, training_history
    
    try:
        print(f"Starting {algorithm_name} training in background thread")
        
        # Filter parameters based on algorithm type
        filtered_params = {}
        
        if algorithm_name in ['q-learning', 'sarsa', 'td-learning', 'nstep-td']:
            # TD methods
            filtered_params = {
                'episodes': params.get('episodes', 1000),
                'gamma': params.get('gamma', 0.99),
                'alpha': params.get('alpha', 0.1),
                'epsilon': params.get('epsilon', 0.1)
            }
            if algorithm_name == 'nstep-td':
                filtered_params['n_steps'] = params.get('n_steps', 3)
                
        elif algorithm_name in ['value-iteration', 'policy-iteration']:
            # DP methods
            filtered_params = {
                'gamma': params.get('gamma', 0.99),
                'theta': 1e-6,
                'max_iterations': params.get('episodes', 100)
            }
            
        elif algorithm_name == 'monte-carlo':
            # Monte Carlo
            filtered_params = {
                'episodes': params.get('episodes', 2000),
                'gamma': params.get('gamma', 0.99),
                'epsilon': params.get('epsilon', 0.1)
            }
        
        print(f"Filtered parameters for {algorithm_name}: {filtered_params}")
        
        # Get algorithm method
        algorithm_method = rl.get_algorithm_method(algorithm_name)
        if not algorithm_method:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Run the algorithm
        result, history = algorithm_method(return_history=True, **filtered_params)
        
        # Store training history
        training_history = list(zip(history.get('episodes', []), history.get('rewards', [])))
        
        # Store results
        training_results = {
            'algorithm': algorithm_name,
            'value_function': result['value_function'],
            'policy': result['policy'],
            'history': history,
            'parameters': filtered_params,
            'trained_at': datetime.now().isoformat(),
            'env_size': rl.size
        }
        
        training_progress = 100
        print(f"{algorithm_name} completed successfully")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training_active = False

@app.route('/api/training_status')
def training_status():
    global current_algorithm, training_progress, training_episode, training_results, training_active, training_history
    
    if not current_algorithm:
        return jsonify({
            'algorithm': 'None',
            'status': 'idle',
            'progress': 0,
            'is_training': False,
            'history': {},
            'results': None
        })
    
    total_episodes = current_algorithm.get('parameters', {}).get('episodes', 1000)
    
    # Prepare chart data
    chart_data = {'episodes': [], 'rewards': []}
    
    if training_results and 'history' in training_results:
        history = training_results['history']
        if 'episodes' in history and 'rewards' in history:
            chart_data['episodes'] = history['episodes']
            chart_data['rewards'] = history['rewards']
    
    # If we have training history from ongoing training
    elif training_history:
        for ep, reward in training_history:
            chart_data['episodes'].append(ep)
            chart_data['rewards'].append(reward)
    
    return jsonify({
        'algorithm': current_algorithm['name'],
        'status': 'training' if training_active else 'completed',
        'current_episode': training_episode,
        'total_episodes': total_episodes,
        'progress': training_progress,
        'is_training': training_active,
        'history': chart_data,
        'results': training_results if training_results else None
    })

@app.route('/api/get_training_results')
def get_training_results():
    global training_results
    if training_results:
        return jsonify(training_results)
    return jsonify({'error': 'No training results available'}), 404

@app.route('/api/run_inference', methods=['POST'])
def run_inference():
    global training_results, training_environment
    
    data = request.json
    start_state = data.get('start_state', [0,0])
    
    if not training_results or 'policy' not in training_results:
        return jsonify({'error': 'No trained model available. Train an algorithm first!'}), 400
    
    if training_environment is None:
        return jsonify({'error': 'Environment not available'}), 400
    
    try:
        # Use the SAME environment instance that was used for training
        env = training_environment
        size = env.size
        policy = training_results['policy']
        
        position = start_state.copy()
        trajectory = [position.copy()]
        total_reward = 0
        steps = 0
        max_steps = size * size * 3
        success = False
        
        print(f"Starting inference from {start_state} with policy size: {len(policy)}x{len(policy[0])}")
        print(f"Environment walls: {env.walls}")
        print(f"Environment goal: {env.goal_state}")
        
        # Reset environment but keep the same instance
        env.reset()
        
        # Follow the policy
        while steps < max_steps:
            row, col = position
            
            # Check if we're at goal
            if position == env.goal_state:
                total_reward += env.goal_reward
                success = True
                trajectory.append(position.copy())
                break
            
            # Check if we're at a wall
            if position in env.walls:
                total_reward += env.wall_reward
                print(f"Hit wall at {position}")
                break
            
            # Get action from policy
            if 0 <= row < len(policy) and 0 <= col < len(policy[0]):
                action = policy[row][col]
                
                # Take the action using the SAME environment
                next_pos, reward, done = env.step(action)
                
                print(f"Step {steps}: Position {position}, Action {action} -> Next {next_pos}, Reward {reward}")
                
                position = next_pos
                total_reward += reward
                trajectory.append(position.copy())
                steps += 1
                
                if done:
                    success = True
                    break
            else:
                print(f"Position {position} out of bounds")
                break
        
        print(f"Inference complete: steps={steps}, success={success}, reward={total_reward}")
        
        return jsonify({
            'trajectory': trajectory, 
            'total_reward': round(total_reward, 2), 
            'steps': steps, 
            'success': success,
            'algorithm': training_results.get('algorithm', 'unknown'),
            'path_length': len(trajectory) - 1
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

@app.route('/api/debug_policy', methods=['POST'])
def debug_policy():
    global training_results, training_environment
    
    if not training_results or 'policy' not in training_results:
        return jsonify({'error': 'No trained policy'}), 400
    
    if training_environment is None:
        return jsonify({'error': 'Environment not available'}), 400
    
    policy = training_results['policy']
    
    # Convert policy numbers to direction symbols
    direction_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    policy_grid = []
    for i in range(len(policy)):
        row = []
        for j in range(len(policy[i])):
            cell = [i, j]
            if cell == training_environment.goal_state:
                row.append('G')
            elif cell in training_environment.walls:
                row.append('█')
            elif cell == training_environment.start_state:
                row.append('S')
            else:
                action = policy[i][j]
                row.append(direction_map.get(action, '?'))
        policy_grid.append(row)
    
    return jsonify({
        'policy_grid': policy_grid,
        'policy': policy,
        'size': len(policy),
        'walls': training_environment.walls,
        'start': training_environment.start_state,
        'goal': training_environment.goal_state
    })

@app.route('/api/get_visualization')
def get_visualization():
    global training_results
    if not training_results:
        return jsonify({'error': 'No training results'}), 404
    
    visualizations = {}
    if 'value_function' in training_results: 
        visualizations['value_function'] = training_results['value_function']
    if 'policy' in training_results: 
        visualizations['policy'] = training_results['policy']
    if 'history' in training_results: 
        visualizations['history'] = training_results['history']
    
    return jsonify(visualizations)

@app.route('/api/init_gridworld', methods=['POST'])
def init_gridworld():
    global training_environment
    
    data = request.json
    size = data.get('size', 5)
    random_walls = data.get('random_walls', True)
    wall_density = data.get('wall_density', 0.15)
    
    try:
        # Create new environment
        training_environment = GridWorld(size=size, random_walls=random_walls, wall_density=wall_density)
        
        # Store environment info in session (NOT the object)
        session['gridworld_env'] = {
            'size': size,
            'start': training_environment.start_state,
            'goal': training_environment.goal_state,
            'walls': training_environment.walls,
            'agent_position': training_environment.start_state,
            'wall_count': len(training_environment.walls),
            'goal_reward': training_environment.goal_reward,
            'step_reward': training_environment.step_reward,
            'wall_reward': training_environment.wall_reward
        }
        
        # Reset training state
        global current_algorithm, training_results
        current_algorithm = None
        training_results = {}
        
        return jsonify({
            'size': size,
            'start': training_environment.start_state,
            'goal': training_environment.goal_state,
            'walls': training_environment.walls,
            'agent_position': training_environment.start_state,
            'wall_count': len(training_environment.walls),
            'goal_reward': training_environment.goal_reward,
            'step_reward': training_environment.step_reward,
            'wall_reward': training_environment.wall_reward
        })
    except Exception as e:
        print(f"Error creating gridworld: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'size': 5,
            'start': [0, 0],
            'goal': [4, 4],
            'walls': [[1, 2], [2, 2], [3, 1]],
            'agent_position': [0, 0],
            'wall_count': 3,
            'goal_reward': 10,
            'step_reward': -0.1,
            'wall_reward': -1
        })

@app.route('/api/move_agent', methods=['POST'])
def move_agent():
    global training_environment
    
    data = request.json
    action = data.get('action')
    
    if training_environment is None:
        return jsonify({'error': 'Environment not initialized'}), 400
    
    # Get current position from environment
    current_pos = training_environment.state.copy()
    
    # Calculate new position
    new_pos = current_pos.copy()
    if action == 0:  # Up
        new_pos[0] = max(0, new_pos[0] - 1)
    elif action == 1:  # Down
        new_pos[0] = min(training_environment.size - 1, new_pos[0] + 1)
    elif action == 2:  # Left
        new_pos[1] = max(0, new_pos[1] - 1)
    elif action == 3:  # Right
        new_pos[1] = min(training_environment.size - 1, new_pos[1] + 1)
    
    # Check for walls
    is_wall = new_pos in training_environment.walls
    if is_wall:
        new_pos = current_pos
    
    reached_goal = new_pos == training_environment.goal_state
    
    # Update environment state
    training_environment.state = new_pos.copy()
    
    # Update session
    session['gridworld_env']['agent_position'] = new_pos
    
    return jsonify({
        'new_position': new_pos,
        'reached_goal': reached_goal,
        'is_wall': is_wall
    })

@app.route('/api/reset_agent')
def reset_agent():
    global training_environment
    
    if training_environment:
        training_environment.reset()
        session['gridworld_env']['agent_position'] = training_environment.start_state.copy()
        return jsonify({'position': training_environment.start_state.copy()})
    
    return jsonify({'position': [0, 0]})

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    global training_active
    training_active = False
    return jsonify({'status': 'stopped', 'message': 'Training stopped'})

@app.route('/api/reset_training', methods=['POST'])
def reset_training():
    global training_active, training_results, training_environment, current_algorithm, training_progress, training_episode, training_history
    
    training_active = False
    training_results = {}
    current_algorithm = None
    training_progress = 0
    training_episode = 0
    training_history = []
    # Keep training_environment for consistency
    
    # Clear session data
    if 'gridworld_env' in session:
        session['gridworld_env']['agent_position'] = session['gridworld_env']['start']
    
    return jsonify({'status': 'reset', 'message': 'Training state reset'})

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})
# FrozenLake routes
frozenlake_algorithm = None
frozenlake_training_active = False
frozenlake_training_results = {}
frozenlake_training_progress = 0
frozenlake_training_episode = 0
frozenlake_training_thread = None
frozenlake_training_history = []
frozenlake_environment = None

@app.route('/frozenlake')
def frozenlake():
    return render_template('frozenlake.html')

@app.route('/api/init_frozenlake', methods=['POST'])
def init_frozenlake():
    global frozenlake_environment
    
    data = request.json
    size = data.get('size', 4)
    is_slippery = data.get('is_slippery', True)
    hole_prob = data.get('hole_prob', 0.2)
    
    try:
        frozenlake_environment = FrozenLake(
            size=size, 
            is_slippery=is_slippery, 
            hole_prob=hole_prob
        )
        
        session['frozenlake_env'] = {
            'size': size,
            'start': frozenlake_environment.start_state,
            'goal': frozenlake_environment.goal_state,
            'holes': frozenlake_environment.holes,
            'agent_position': frozenlake_environment.start_state,
            'hole_count': len(frozenlake_environment.holes),
            'is_slippery': is_slippery,
            'goal_reward': frozenlake_environment.goal_reward,
            'hole_reward': frozenlake_environment.hole_reward,
            'step_reward': frozenlake_environment.step_reward
        }
        
        global frozenlake_algorithm, frozenlake_training_results
        frozenlake_algorithm = None
        frozenlake_training_results = {}
        
        return jsonify({
            'size': size,
            'start': frozenlake_environment.start_state,
            'goal': frozenlake_environment.goal_state,
            'holes': frozenlake_environment.holes,
            'agent_position': frozenlake_environment.start_state,
            'hole_count': len(frozenlake_environment.holes),
            'is_slippery': is_slippery,
            'goal_reward': frozenlake_environment.goal_reward,
            'hole_reward': frozenlake_environment.hole_reward,
            'step_reward': frozenlake_environment.step_reward
        })
    except Exception as e:
        print(f"Error creating frozenlake: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'size': 4,
            'start': [0, 0],
            'goal': [3, 3],
            'holes': [[1, 1], [2, 3]],
            'agent_position': [0, 0],
            'hole_count': 2,
            'is_slippery': True,
            'goal_reward': 10,
            'hole_reward': -10,
            'step_reward': -0.1
        })


@app.route('/api/train_frozenlake', methods=['POST'])
def train_frozenlake():
    global frozenlake_training_active, frozenlake_algorithm, frozenlake_training_progress, frozenlake_training_episode, frozenlake_training_results, frozenlake_training_thread, frozenlake_training_history, frozenlake_environment
    
    if frozenlake_training_active:
        return jsonify({'error': 'Training already in progress'}), 400

    data = request.json
    algorithm_name = data.get('algorithm')
    params = data.get('parameters', {})
    is_slippery = data.get('is_slippery', True)
    
    size = 4
    
    try:
        if frozenlake_environment is None:
            print(f"Creating FrozenLake environment with size={size}, slippery={is_slippery}")
            frozenlake_environment = FrozenLake(size=size, is_slippery=is_slippery, hole_prob=0.2)
        else:
            # Update slippery state if different
            if frozenlake_environment.is_slippery != is_slippery:
                frozenlake_environment.is_slippery = is_slippery
        
        rl = RLAlgorithm(frozenlake_environment)
        
        frozenlake_algorithm = {
            'name': algorithm_name,
            'parameters': params,
            'status': 'training',
            'started_at': datetime.now().isoformat(),
            'progress': 0,
            'env_size': size,
            'is_slippery': is_slippery
        }
        
        frozenlake_training_progress = 0
        frozenlake_training_episode = 0
        frozenlake_training_active = True
        frozenlake_training_history = []
        
        session['frozenlake_env'] = {
            'size': frozenlake_environment.size,
            'start': frozenlake_environment.start_state,
            'goal': frozenlake_environment.goal_state,
            'holes': frozenlake_environment.holes,
            'agent_position': frozenlake_environment.start_state.copy(),
            'hole_count': len(frozenlake_environment.holes),
            'is_slippery': is_slippery,
            'goal_reward': frozenlake_environment.goal_reward,
            'hole_reward': frozenlake_environment.hole_reward,
            'step_reward': frozenlake_environment.step_reward
        }
        
        session_id = f"frozenlake_session_{int(time.time())}"
        print(f"Starting training thread for {algorithm_name} with session_id: {session_id}")
        
        frozenlake_training_thread = threading.Thread(
            target=run_frozenlake_training_thread, 
            args=(session_id, rl, algorithm_name, params)
        )
        frozenlake_training_thread.daemon = True
        frozenlake_training_thread.start()
        
        return jsonify({
            'status': 'started', 
            'algorithm': algorithm_name, 
            'session_id': session_id,
            'message': f'Training {algorithm_name} started'
        })
        
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        traceback.print_exc()
        frozenlake_training_active = False
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

def run_frozenlake_training_thread(session_id, rl, algorithm_name, params):
    """Background thread for FrozenLake training"""
    global frozenlake_training_active, frozenlake_training_results, frozenlake_training_progress, frozenlake_training_episode, frozenlake_training_history
    
    try:
        print(f"Starting {algorithm_name} training in background thread")
        
        # Get algorithm method
        algorithm_method = rl.get_algorithm_method(algorithm_name)
        if not algorithm_method:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Filter parameters based on algorithm type
        filtered_params = {}
        
        if algorithm_name in ['q-learning', 'sarsa', 'td-learning', 'nstep-td']:
            # TD methods
            filtered_params = {
                'episodes': params.get('episodes', 2000),
                'gamma': params.get('gamma', 0.99),
                'alpha': params.get('alpha', 0.1),
                'epsilon': params.get('epsilon', 0.1)
            }
            if algorithm_name == 'nstep-td':
                filtered_params['n_steps'] = params.get('n_steps', 3)
                
        elif algorithm_name in ['value-iteration', 'policy-iteration']:
            # DP methods
            filtered_params = {
                'gamma': params.get('gamma', 0.99),
                'theta': 1e-6,
                'max_iterations': params.get('episodes', 100)  # Use episodes as max_iterations
            }
            
        elif algorithm_name == 'monte-carlo':
            # Monte Carlo
            filtered_params = {
                'episodes': params.get('episodes', 2000),
                'gamma': params.get('gamma', 0.99),
                'epsilon': params.get('epsilon', 0.1)
            }
        
        print(f"Filtered parameters for {algorithm_name}: {filtered_params}")
        
        # Run the algorithm with filtered parameters
        result, history = algorithm_method(return_history=True, **filtered_params)
        
        frozenlake_training_history = list(zip(history.get('episodes', []), history.get('rewards', [])))
        
        frozenlake_training_results = {
            'algorithm': algorithm_name,
            'value_function': result['value_function'],
            'policy': result['policy'],
            'history': history,
            'parameters': filtered_params,
            'trained_at': datetime.now().isoformat(),
            'env_size': rl.size
        }
        
        frozenlake_training_progress = 100
        print(f"{algorithm_name} completed successfully")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        frozenlake_training_active = False

def monte_carlo_first_visit(env, episodes=2000, gamma=0.99, epsilon=0.1):
    """Monte Carlo First-Visit control algorithm for FrozenLake"""
    print(f"Starting Monte Carlo First-Visit with {episodes} episodes")
    
    size = env.size
    Q = np.random.randn(size, size, 4) * 0.01
    returns = {(i, j, a): [] for i in range(size) for j in range(size) for a in range(4)}
    history = {'episodes': [], 'rewards': [], 'steps': []}
    
    for ep in range(episodes):
        state = env.reset()
        episode = []
        total_reward = 0
        steps = 0
        done = False
        
        # Generate episode
        while not done and steps < 200:
            x, y = state
            
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                valid_actions = env.get_available_actions(state)
                action = np.random.choice(valid_actions)
            else:
                action = np.argmax(Q[x, y])
            
            next_state, reward, done = env.step(action)
            
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
                Q[x, y, action] = np.mean(returns[(x, y, action)])
        
        # Record history
        if ep % 10 == 0 or ep == episodes-1:
            history['episodes'].append(ep)
            history['rewards'].append(total_reward)
            history['steps'].append(steps)
        
        if ep % 200 == 0:
            print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    # Extract policy
    policy = np.argmax(Q, axis=2)
    value_function = np.max(Q, axis=2)
    
    result = {
        'Q': Q.tolist(),
        'policy': policy.tolist(),
        'value_function': value_function.tolist(),
        'algorithm': 'monte-carlo',
        'parameters': {'episodes': episodes, 'gamma': gamma, 'epsilon': epsilon}
    }
    
    return result, history

@app.route('/api/frozenlake_training_status')
def frozenlake_training_status():
    global frozenlake_algorithm, frozenlake_training_progress, frozenlake_training_episode, frozenlake_training_results, frozenlake_training_active, frozenlake_training_history
    
    if not frozenlake_algorithm:
        return jsonify({
            'algorithm': 'None',
            'status': 'idle',
            'progress': 0,
            'is_training': False,
            'history': {},
            'results': None
        })
    
    total_episodes = frozenlake_algorithm.get('parameters', {}).get('episodes', 2000)
    
    chart_data = {'episodes': [], 'rewards': []}
    
    if frozenlake_training_results and 'history' in frozenlake_training_results:
        history = frozenlake_training_results['history']
        if 'episodes' in history and 'rewards' in history:
            chart_data['episodes'] = history['episodes']
            chart_data['rewards'] = history['rewards']
    
    elif frozenlake_training_history:
        for ep, reward in frozenlake_training_history:
            chart_data['episodes'].append(ep)
            chart_data['rewards'].append(reward)
    
    return jsonify({
        'algorithm': frozenlake_algorithm['name'],
        'status': 'training' if frozenlake_training_active else 'completed',
        'current_episode': frozenlake_training_episode,
        'total_episodes': total_episodes,
        'progress': frozenlake_training_progress,
        'is_training': frozenlake_training_active,
        'history': chart_data,
        'results': frozenlake_training_results if frozenlake_training_results else None
    })

@app.route('/api/run_frozenlake_inference', methods=['POST'])
def run_frozenlake_inference():
    global frozenlake_training_results, frozenlake_environment
    
    data = request.json
    start_state = data.get('start_state', [0,0])
    
    if not frozenlake_training_results or 'policy' not in frozenlake_training_results:
        return jsonify({'error': 'No trained model available. Train an algorithm first!'}), 400
    
    if frozenlake_environment is None:
        return jsonify({'error': 'Environment not available'}), 400
    
    try:
        env = frozenlake_environment
        size = env.size
        policy = frozenlake_training_results['policy']
        
        position = start_state.copy()
        trajectory = [position.copy()]
        total_reward = 0
        steps = 0
        max_steps = size * size * 5
        success = False
        
        print(f"Starting inference from {start_state} with policy size: {len(policy)}x{len(policy[0])}")
        print(f"Environment holes: {env.holes}")
        print(f"Environment goal: {env.goal_state}")
        
        env.reset()
        
        while steps < max_steps:
            row, col = position
            
            if position == env.goal_state:
                total_reward += env.goal_reward
                success = True
                trajectory.append(position.copy())
                break
            
            if position in env.holes:
                total_reward += env.hole_reward
                print(f"Fell in hole at {position}")
                break
            
            if 0 <= row < len(policy) and 0 <= col < len(policy[0]):
                action = policy[row][col]
                
                next_pos, reward, done = env.step(action)
                
                print(f"Step {steps}: Position {position}, Action {action} -> Next {next_pos}, Reward {reward}")
                
                position = next_pos
                total_reward += reward
                trajectory.append(position.copy())
                steps += 1
                
                if done:
                    success = True
                    break
            else:
                print(f"Position {position} out of bounds")
                break
        
        print(f"Inference complete: steps={steps}, success={success}, reward={total_reward}")
        
        return jsonify({
            'trajectory': trajectory, 
            'total_reward': round(total_reward, 2), 
            'steps': steps, 
            'success': success,
            'algorithm': frozenlake_training_results.get('algorithm', 'unknown'),
            'path_length': len(trajectory) - 1
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

@app.route('/api/debug_frozenlake_policy', methods=['POST'])
def debug_frozenlake_policy():
    global frozenlake_training_results, frozenlake_environment
    
    if not frozenlake_training_results or 'policy' not in frozenlake_training_results:
        return jsonify({'error': 'No trained policy'}), 400
    
    if frozenlake_environment is None:
        return jsonify({'error': 'Environment not available'}), 400
    
    policy = frozenlake_training_results['policy']
    
    direction_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    policy_grid = []
    for i in range(len(policy)):
        row = []
        for j in range(len(policy[i])):
            cell = [i, j]
            if cell == frozenlake_environment.goal_state:
                row.append('G')
            elif cell in frozenlake_environment.holes:
                row.append('H')
            elif cell == frozenlake_environment.start_state:
                row.append('S')
            else:
                action = policy[i][j]
                row.append(direction_map.get(action, '?'))
        policy_grid.append(row)
    
    return jsonify({
        'policy_grid': policy_grid,
        'policy': policy,
        'size': len(policy),
        'holes': frozenlake_environment.holes,
        'start': frozenlake_environment.start_state,
        'goal': frozenlake_environment.goal_state,
        'is_slippery': frozenlake_environment.is_slippery
    })




# CliffWalking routes
cliffwalking_algorithm = None
cliffwalking_training_active = False
cliffwalking_training_results = {}
cliffwalking_training_progress = 0
cliffwalking_training_episode = 0
cliffwalking_training_thread = None
cliffwalking_training_history = []
cliffwalking_environment = None

@app.route('/cliffwalking')
def cliffwalking():
    return render_template('cliffwalking.html')

@app.route('/api/init_cliffwalking', methods=['POST'])
def init_cliffwalking():
    global cliffwalking_environment
    
    data = request.json
    width = data.get('width', 12)
    height = data.get('height', 4)
    
    try:
        cliffwalking_environment = CliffWalking(width=width, height=height)
        
        session['cliffwalking_env'] = {
            'width': width,
            'height': height,
            'start': cliffwalking_environment.start_state,
            'goal': cliffwalking_environment.goal_state,
            'cliff_cells': cliffwalking_environment.cliff_cells,
            'agent_position': cliffwalking_environment.start_state,
            'cliff_count': len(cliffwalking_environment.cliff_cells),
            'goal_reward': cliffwalking_environment.goal_reward,
            'cliff_reward': cliffwalking_environment.cliff_reward,
            'step_reward': cliffwalking_environment.step_reward
        }
        
        global cliffwalking_algorithm, cliffwalking_training_results
        cliffwalking_algorithm = None
        cliffwalking_training_results = {}
        
        return jsonify({
            'width': width,
            'height': height,
            'start': cliffwalking_environment.start_state,
            'goal': cliffwalking_environment.goal_state,
            'cliff_cells': cliffwalking_environment.cliff_cells,
            'agent_position': cliffwalking_environment.start_state,
            'cliff_count': len(cliffwalking_environment.cliff_cells),
            'goal_reward': cliffwalking_environment.goal_reward,
            'cliff_reward': cliffwalking_environment.cliff_reward,
            'step_reward': cliffwalking_environment.step_reward
        })
    except Exception as e:
        print(f"Error creating cliffwalking: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'width': 12,
            'height': 4,
            'start': [3, 0],
            'goal': [3, 11],
            'cliff_cells': [[3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10]],
            'agent_position': [3, 0],
            'cliff_count': 10,
            'goal_reward': 10,
            'cliff_reward': -100,
            'step_reward': -1
        })


@app.route('/api/get_frozenlake_training_results')
def get_frozenlake_training_results():
    global frozenlake_training_results
    if frozenlake_training_results:
        return jsonify(frozenlake_training_results)

    return jsonify({'error': 'No training results available'}), 404
@app.route('/api/train_cliffwalking', methods=['POST'])
def train_cliffwalking():
    global cliffwalking_training_active, cliffwalking_algorithm, cliffwalking_training_progress, cliffwalking_training_episode, cliffwalking_training_results, cliffwalking_training_thread, cliffwalking_training_history, cliffwalking_environment
    
    if cliffwalking_training_active:
        return jsonify({'error': 'Training already in progress'}), 400

    data = request.json
    algorithm_name = data.get('algorithm')
    params = data.get('parameters', {})
    size_type = data.get('size_type', 'small')
    
    # Set dimensions based on size type
    if size_type == 'small':
        width, height = 12, 4
    elif size_type == 'medium':
        width, height = 15, 5
    elif size_type == 'large':
        width, height = 20, 6
    else:
        width, height = 12, 4
    
    try:
        if cliffwalking_environment is None:
            print(f"Creating CliffWalking environment with width={width}, height={height}")
            cliffwalking_environment = CliffWalking(width=width, height=height)
        else:
            # Recreate environment if dimensions changed
            if cliffwalking_environment.width != width or cliffwalking_environment.height != height:
                cliffwalking_environment = CliffWalking(width=width, height=height)
        
        rl = RLAlgorithm(cliffwalking_environment)
        
        cliffwalking_algorithm = {
            'name': algorithm_name,
            'parameters': params,
            'status': 'training',
            'started_at': datetime.now().isoformat(),
            'progress': 0,
            'width': width,
            'height': height
        }
        
        cliffwalking_training_progress = 0
        cliffwalking_training_episode = 0
        cliffwalking_training_active = True
        cliffwalking_training_history = []
        
        session['cliffwalking_env'] = {
            'width': width,
            'height': height,
            'start': cliffwalking_environment.start_state,
            'goal': cliffwalking_environment.goal_state,
            'cliff_cells': cliffwalking_environment.cliff_cells,
            'agent_position': cliffwalking_environment.start_state.copy(),
            'cliff_count': len(cliffwalking_environment.cliff_cells),
            'goal_reward': cliffwalking_environment.goal_reward,
            'cliff_reward': cliffwalking_environment.cliff_reward,
            'step_reward': cliffwalking_environment.step_reward
        }
        
        session_id = f"cliffwalking_session_{int(time.time())}"
        print(f"Starting training thread for {algorithm_name} with session_id: {session_id}")
        
        cliffwalking_training_thread = threading.Thread(
            target=run_cliffwalking_training_thread, 
            args=(session_id, rl, algorithm_name, params)
        )
        cliffwalking_training_thread.daemon = True
        cliffwalking_training_thread.start()
        
        return jsonify({
            'status': 'started', 
            'algorithm': algorithm_name, 
            'session_id': session_id,
            'message': f'Training {algorithm_name} started'
        })
        
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        traceback.print_exc()
        cliffwalking_training_active = False
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

def run_cliffwalking_training_thread(session_id, rl, algorithm_name, params):
    """Background thread for CliffWalking training"""
    global cliffwalking_training_active, cliffwalking_training_results, cliffwalking_training_progress, cliffwalking_training_episode, cliffwalking_training_history
    
    try:
        print(f"Starting {algorithm_name} training in background thread")
        
        # Filter parameters based on algorithm type
        filtered_params = {}
        
        if algorithm_name in ['q-learning', 'sarsa', 'td-learning', 'nstep-td']:
            # TD methods
            filtered_params = {
                'episodes': params.get('episodes', 2000),
                'gamma': params.get('gamma', 0.99),
                'alpha': params.get('alpha', 0.1),
                'epsilon': params.get('epsilon', 0.1)
            }
            if algorithm_name == 'nstep-td':
                filtered_params['n_steps'] = params.get('n_steps', 3)
                
        elif algorithm_name in ['value-iteration', 'policy-iteration']:
            # DP methods
            filtered_params = {
                'gamma': params.get('gamma', 0.99),
                'theta': 1e-6,
                'max_iterations': params.get('episodes', 100)
            }
            
        elif algorithm_name == 'monte-carlo':
            # Monte Carlo
            filtered_params = {
                'episodes': params.get('episodes', 2000),
                'gamma': params.get('gamma', 0.99),
                'epsilon': params.get('epsilon', 0.1)
            }
        
        print(f"Filtered parameters for {algorithm_name}: {filtered_params}")
        
        # Get algorithm method
        algorithm_method = rl.get_algorithm_method(algorithm_name)
        if not algorithm_method:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Run the algorithm
        result, history = algorithm_method(return_history=True, **filtered_params)
        
        cliffwalking_training_history = list(zip(history.get('episodes', []), history.get('rewards', [])))
        
        cliffwalking_training_results = {
            'algorithm': algorithm_name,
            'value_function': result['value_function'],
            'policy': result['policy'],
            'history': history,
            'parameters': filtered_params,
            'trained_at': datetime.now().isoformat(),
            'width': rl.env.width,
            'height': rl.env.height
        }
        
        cliffwalking_training_progress = 100
        print(f"{algorithm_name} completed successfully")
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cliffwalking_training_active = False

@app.route('/api/cliffwalking_training_status')
def cliffwalking_training_status():
    global cliffwalking_algorithm, cliffwalking_training_progress, cliffwalking_training_episode, cliffwalking_training_results, cliffwalking_training_active, cliffwalking_training_history
    
    if not cliffwalking_algorithm:
        return jsonify({
            'algorithm': 'None',
            'status': 'idle',
            'progress': 0,
            'is_training': False,
            'history': {},
            'results': None
        })
    
    total_episodes = cliffwalking_algorithm.get('parameters', {}).get('episodes', 2000)
    
    chart_data = {'episodes': [], 'rewards': []}
    
    if cliffwalking_training_results and 'history' in cliffwalking_training_results:
        history = cliffwalking_training_results['history']
        if 'episodes' in history and 'rewards' in history:
            chart_data['episodes'] = history['episodes']
            chart_data['rewards'] = history['rewards']
    
    elif cliffwalking_training_history:
        for ep, reward in cliffwalking_training_history:
            chart_data['episodes'].append(ep)
            chart_data['rewards'].append(reward)
    
    return jsonify({
        'algorithm': cliffwalking_algorithm['name'],
        'status': 'training' if cliffwalking_training_active else 'completed',
        'current_episode': cliffwalking_training_episode,
        'total_episodes': total_episodes,
        'progress': cliffwalking_training_progress,
        'is_training': cliffwalking_training_active,
        'history': chart_data,
        'results': cliffwalking_training_results if cliffwalking_training_results else None
    })

@app.route('/api/run_cliffwalking_inference', methods=['POST'])
def run_cliffwalking_inference():
    global cliffwalking_training_results, cliffwalking_environment
    
    data = request.json
    start_state = data.get('start_state', [3, 0])
    
    if not cliffwalking_training_results or 'policy' not in cliffwalking_training_results:
        return jsonify({'error': 'No trained model available. Train an algorithm first!'}), 400
    
    if cliffwalking_environment is None:
        return jsonify({'error': 'Environment not available'}), 400
    
    try:
        env = cliffwalking_environment
        width = env.width
        height = env.height
        policy = cliffwalking_training_results['policy']
        
        position = start_state.copy()
        trajectory = [position.copy()]
        total_reward = 0
        steps = 0
        max_steps = width * height * 3
        success = False
        
        print(f"Starting inference from {start_state} with policy size: {len(policy)}x{len(policy[0])}")
        print(f"Environment cliff cells: {env.cliff_cells}")
        print(f"Environment goal: {env.goal_state}")
        
        env.reset()
        
        while steps < max_steps:
            row, col = position
            
            if position == env.goal_state:
                total_reward += env.goal_reward
                success = True
                trajectory.append(position.copy())
                break
            
            if position in env.cliff_cells:
                total_reward += env.cliff_reward
                print(f"Fell off cliff at {position}")
                # Reset to start after falling
                position = env.start_state.copy()
                trajectory.append(position.copy())
                continue
            
            if 0 <= row < len(policy) and 0 <= col < len(policy[0]):
                action = policy[row][col]
                
                next_pos, reward, done = env.step(action)
                
                print(f"Step {steps}: Position {position}, Action {action} -> Next {next_pos}, Reward {reward}")
                
                position = next_pos
                total_reward += reward
                trajectory.append(position.copy())
                steps += 1
                
                if done:
                    success = True
                    break
            else:
                print(f"Position {position} out of bounds")
                break
        
        print(f"Inference complete: steps={steps}, success={success}, reward={total_reward}")
        
        return jsonify({
            'trajectory': trajectory, 
            'total_reward': round(total_reward, 2), 
            'steps': steps, 
            'success': success,
            'algorithm': cliffwalking_training_results.get('algorithm', 'unknown'),
            'path_length': len(trajectory) - 1
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

@app.route('/api/debug_cliffwalking_policy', methods=['POST'])
def debug_cliffwalking_policy():
    global cliffwalking_training_results, cliffwalking_environment
    
    if not cliffwalking_training_results or 'policy' not in cliffwalking_training_results:
        return jsonify({'error': 'No trained policy'}), 400
    
    if cliffwalking_environment is None:
        return jsonify({'error': 'Environment not available'}), 400
    
    policy = cliffwalking_training_results['policy']
    
    direction_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    policy_grid = []
    for i in range(len(policy)):
        row = []
        for j in range(len(policy[i])):
            cell = [i, j]
            if cell == cliffwalking_environment.goal_state:
                row.append('G')
            elif cell in cliffwalking_environment.cliff_cells:
                row.append('C')
            elif cell == cliffwalking_environment.start_state:
                row.append('S')
            else:
                action = policy[i][j]
                row.append(direction_map.get(action, '?'))
        policy_grid.append(row)
    
    return jsonify({
        'policy_grid': policy_grid,
        'policy': policy,
        'width': cliffwalking_environment.width,
        'height': cliffwalking_environment.height,
        'cliff_cells': cliffwalking_environment.cliff_cells,
        'start': cliffwalking_environment.start_state,
        'goal': cliffwalking_environment.goal_state
    })



if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, port=5000, threaded=True)