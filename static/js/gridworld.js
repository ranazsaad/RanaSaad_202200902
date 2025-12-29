// Global variables
let currentSessionId = null;
let gridVisualizer = null;
let trainingChart = null;
let progressInterval = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('GridWorld page loaded');
    
    // Initialize visualizers
    initVisualizers();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize grid
    initGrid();
});

function initVisualizers() {
    // Create grid canvas
    const canvas = document.getElementById('grid-canvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }
    
    // Initialize training chart
    const chartCanvas = document.getElementById('training-chart');
    if (chartCanvas) {
        trainingChart = new Chart(chartCanvas.getContext('2d'), {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Reward per Episode',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Episode' }
                    },
                    y: {
                        title: { display: true, text: 'Reward' }
                    }
                }
            }
        });
    }
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
}

function setupEventListeners() {
    // Grid size slider
    const gridSizeSlider = document.getElementById('grid-size');
    if (gridSizeSlider) {
        gridSizeSlider.addEventListener('input', function() {
            const size = this.value;
            const display = document.getElementById('grid-size-value');
            if (display) display.textContent = `${size}x${size}`;
        });
    }
    
    // Parameter sliders
    ['gamma', 'alpha', 'epsilon', 'episodes'].forEach(param => {
        const slider = document.getElementById(param);
        const display = document.getElementById(`${param}-value`);
        if (slider && display) {
            slider.addEventListener('input', function() {
                display.textContent = this.value;
            });
        }
    });
    
    // Algorithm selection
    document.querySelectorAll('.algo-card .btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const algorithm = this.getAttribute('data-algorithm');
            selectAlgorithm(algorithm);
        });
    });
    
    // Start training button
    const startBtn = document.getElementById('start-training');
    if (startBtn) {
        startBtn.addEventListener('click', startTraining);
    }
    
    // Stop training button
    const stopBtn = document.getElementById('stop-training');
    if (stopBtn) {
        stopBtn.addEventListener('click', stopTraining);
    }
    
    // Run inference button
    const inferenceBtn = document.getElementById('run-inference');
    if (inferenceBtn) {
        inferenceBtn.addEventListener('click', runInference);
    }
    
    // Reset button
    const resetBtn = document.getElementById('reset-training');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetTraining);
    }
}

function selectAlgorithm(algorithm) {
    // Highlight selected algorithm
    document.querySelectorAll('.algo-card').forEach(card => {
        card.classList.remove('selected');
    });
    event.target.closest('.algo-card').classList.add('selected');
    
    // Update algorithm info
    const algoDisplay = document.getElementById('selected-algorithm');
    if (algoDisplay) {
        algoDisplay.textContent = algorithm.charAt(0).toUpperCase() + algorithm.slice(1);
    }
    
    // Enable training button
    const startBtn = document.getElementById('start-training');
    if (startBtn) startBtn.disabled = false;
}
function updateDebugInfo(status) {
    const debugDiv = document.getElementById('debug-info');
    if (!debugDiv) return;
    
    if (!status || !currentAlgorithm) {
        debugDiv.innerHTML = '<span class="badge bg-secondary">No training data</span>';
        return;
    }
    
    const html = `
        <div class="mb-1">
            <span class="badge bg-info">Algorithm: ${status.algorithm}</span>
            <span class="badge ${status.is_training ? 'bg-warning' : 'bg-success'}">
                ${status.is_training ? 'Training...' : 'Ready'}
            </span>
            <span class="badge bg-primary">Episode: ${status.current_episode || 0}/${status.total_episodes || 0}</span>
        </div>
        <div class="mb-1">
            <small>Progress: ${Math.round(status.progress || 0)}%</small>
            ${status.history && status.history.rewards ? 
                `<small> | Last reward: ${status.history.rewards[status.history.rewards.length-1]?.toFixed(2) || 'N/A'}</small>` : 
                ''}
        </div>
    `;
    
    debugDiv.innerHTML = html;
}
async function initGrid() {
    try {
        const response = await fetch('/api/init_gridworld', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ size: 5 })
        });
        
        if (!response.ok) {
            throw new Error('Failed to initialize grid');
        }
        
        const data = await response.json();
        console.log('Grid initialized:', data);
        
        // Render the grid
        renderGrid(data);
        
        showNotification('GridWorld initialized!', 'success');
        
    } catch (error) {
        console.error('Error initializing grid:', error);
        showNotification('Failed to initialize grid', 'error');
    }
}

function renderGrid(gridData) {
    const container = document.getElementById('grid-container');
    if (!container) return;
    
    container.innerHTML = '';
    container.style.display = 'grid';
    container.style.gridTemplateColumns = `repeat(${gridData.size}, 60px)`;
    container.style.gap = '2px';
    container.style.padding = '10px';
    container.style.backgroundColor = '#f8f9fa';
    container.style.borderRadius = '5px';
    
    for (let i = 0; i < gridData.size; i++) {
        for (let j = 0; j < gridData.size; j++) {
            const cell = document.createElement('div');
            cell.style.width = '60px';
            cell.style.height = '60px';
            cell.style.display = 'flex';
            cell.style.alignItems = 'center';
            cell.style.justifyContent = 'center';
            cell.style.border = '1px solid #ddd';
            cell.style.fontWeight = 'bold';
            cell.style.fontSize = '20px';
            
            const isStart = i === gridData.start[0] && j === gridData.start[1];
            const isGoal = i === gridData.goal[0] && j === gridData.goal[1];
            const isAgent = i === gridData.agent_position[0] && j === gridData.agent_position[1];
            const isWall = gridData.walls.some(w => w[0] === i && w[1] === j);
            
            if (isAgent) {
                cell.style.backgroundColor = '#2196F3';
                cell.style.color = 'white';
                cell.innerHTML = '<i class="fas fa-robot"></i>';
            } else if (isStart) {
                cell.style.backgroundColor = '#4CAF50';
                cell.style.color = 'white';
                cell.textContent = 'S';
            } else if (isGoal) {
                cell.style.backgroundColor = '#FF5722';
                cell.style.color = 'white';
                cell.textContent = 'G';
            } else if (isWall) {
                cell.style.backgroundColor = '#333';
                cell.style.color = 'white';
                cell.textContent = '█';
            } else {
                cell.style.backgroundColor = '#fff';
                cell.textContent = '·';
            }
            
            container.appendChild(cell);
        }
    }
    
    // Update agent position display
    const agentPos = document.getElementById('agent-position');
    if (agentPos) {
        agentPos.textContent = `[${gridData.agent_position[0]},${gridData.agent_position[1]}]`;
    }
}
async function startTraining() {
    if (!currentAlgorithm) {
        showNotification('Please select an algorithm first!', 'warning');
        return;
    }
    
    if (isTraining) {
        showNotification('Training already in progress!', 'warning');
        return;
    }
    
    const params = getCurrentParameters();
    
    // Disable start button, enable stop button
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    
    // Reset chart
    trainingChart.data.labels = [];
    trainingChart.data.datasets[0].data = [];
    trainingChart.update();
    
    // Reset progress
    document.getElementById('training-progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    
    isTraining = true;
    document.getElementById('algorithm-status-text').textContent = 
        `${currentAlgorithm.replace('-', ' ').toUpperCase()} Training...`;
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                algorithm: currentAlgorithm,
                parameters: params
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        showNotification(`Training ${currentAlgorithm} started!`, 'success');
        
        // Start progress monitoring
        startProgressMonitoring();
        
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Failed to start training: ' + error.message, 'error');
        stopTrainingUI();
    }
}

async function startProgressMonitoring() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/training_status');
            const status = await response.json();
            
            // Update debug info
            updateDebugInfo(status);
            
            // Rest of your existing code...
            const progress = status.progress || 0;
            document.getElementById('training-progress-bar').style.width = `${progress}%`;
            document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
            
            // ... rest of your existing code
            // Update status text
            document.getElementById('algorithm-status-text').textContent = 
                `${status.algorithm.toUpperCase()} - ${Math.round(progress)}%`;
            
            // Update chart if we have history data
            if (status.history && status.history.rewards && status.history.rewards.length > 0) {
                trainingChart.data.labels = status.history.episodes || 
                    Array.from({length: status.history.rewards.length}, (_, i) => i);
                trainingChart.data.datasets[0].data = status.history.rewards;
                trainingChart.update();
            }
            
            // Check if training is complete
            if (progress >= 100 || !status.is_training) {
                clearInterval(progressInterval);
                stopTrainingUI();
                showNotification('Training completed!', 'success');
                document.getElementById('algorithm-status-text').textContent = 
                    `${currentAlgorithm.replace('-', ' ').toUpperCase()} Complete`;
                
                // Enable inference button
                document.getElementById('btn-inference').disabled = false;
            }
            
        } catch (error) {
            console.error('Error checking progress:', error);
        }
    }, 1000);
}

async function moveAgent(direction) {
    if (!currentGridData) {
        showNotification('Grid not initialized', 'warning');
        return;
    }
    
    let action;
    switch(direction) {
        case 'up': action = 0; break;
        case 'down': action = 1; break;
        case 'left': action = 2; break;
        case 'right': action = 3; break;
        default: return;
    }
    
    try {
        const response = await fetch('/api/move_agent', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ action: action })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update agent position
        currentGridData.agent_position = data.new_position;
        renderGrid(currentGridData);
        updateGridInfo();
        
        if (data.reached_goal) {
            showNotification('🎉 Goal reached! 🎉', 'success');
        }
        
    } catch (error) {
        console.error('Error moving agent:', error);
        showNotification('Failed to move agent', 'error');
    }
}

async function runInference() {
    if (!currentAlgorithm) {
        showNotification('Please train an algorithm first!', 'warning');
        return;
    }
    
    try {
        showNotification('Running inference...', 'info');
        
        const response = await fetch('/api/run_inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                start_state: [0, 0]
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showNotification(data.error, 'error');
            return;
        }
        
        // Display results
        document.getElementById('result-steps').textContent = data.steps;
        document.getElementById('result-reward').textContent = data.total_reward.toFixed(2);
        document.getElementById('result-success').textContent = data.success ? '✓ Yes' : '✗ No';
        
        // Show reward explanation
        showRewardExplanation(data.total_reward, data.steps, data.success);
        
        // Animate the path
        if (data.trajectory && data.trajectory.length > 0) {
            animatePath(data.trajectory);
        }
        
        showNotification('Inference completed!', 'success');
        
    } catch (error) {
        console.error('Error running inference:', error);
        showNotification('Failed to run inference', 'error');
    }
}

function showRewardExplanation(totalReward, steps, success) {
    const explanation = `
        <strong>Reward Breakdown:</strong><br>
        • ${success ? '+10' : '+0'} for reaching goal<br>
        • -${(steps * 0.1).toFixed(1)} for ${steps} steps (-0.1 per step)<br>
        • ${success ? 'Total: ' + totalReward.toFixed(2) : 'Failed to reach goal'}
    `;
    
    // Create a temporary popup or update a div
    const rewardDiv = document.getElementById('reward-explanation');
    if (rewardDiv) {
        rewardDiv.innerHTML = explanation;
        rewardDiv.style.display = 'block';
    }
}
function startProgressMonitoring() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    progressInterval = setInterval(async () => {
        if (!currentSessionId) return;
        
        try {
            const response = await fetch(`/api/progress/${currentSessionId}`);
            const data = await response.json();
            
            if (data.error) {
                clearInterval(progressInterval);
                return;
            }
            
            // Update progress bar
            const progress = data.progress || 0;
            const progressBar = document.getElementById('training-progress');
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${Math.round(progress)}%`;
            }
            
            // Update episode count
            const currentEpisode = document.getElementById('current-episode');
            const totalEpisodes = document.getElementById('total-episodes');
            if (currentEpisode) currentEpisode.textContent = data.current_episode || 0;
            if (totalEpisodes) totalEpisodes.textContent = data.total_episodes || 0;
            
            // Update reward display
            const currentReward = document.getElementById('current-reward');
            if (currentReward) {
                if (data.history && data.history.rewards && data.history.rewards.length > 0) {
                    const lastReward = data.history.rewards[data.history.rewards.length - 1];
                    currentReward.textContent = lastReward.toFixed(2);
                } else {
                    currentReward.textContent = '0.00';
                }
            }
            
            // Update chart
            if (trainingChart && data.history && data.history.rewards && data.history.episodes) {
                trainingChart.data.datasets[0].data = data.history.rewards.map((reward, idx) => ({
                    x: data.history.episodes[idx],
                    y: reward
                }));
                trainingChart.update();
            }
            
            // Check if training is complete
            if (!data.is_training && progress >= 100) {
                clearInterval(progressInterval);
                const startBtn = document.getElementById('start-training');
                const stopBtn = document.getElementById('stop-training');
                if (startBtn) startBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;
                showNotification('Training completed!', 'success');
                
                // Load results
                loadTrainingResults();
            }
            
        } catch (error) {
            console.error('Error fetching progress:', error);
        }
    }, 1000); // Update every second
}

async function stopTraining() {
    if (!currentSessionId) return;
    
    try {
        await fetch(`/api/stop_training/${currentSessionId}`, {
            method: 'POST'
        });
        
        clearInterval(progressInterval);
        const startBtn = document.getElementById('start-training');
        const stopBtn = document.getElementById('stop-training');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
        showNotification('Training stopped', 'warning');
        
    } catch (error) {
        console.error('Error stopping training:', error);
    }
}

async function loadTrainingResults() {
    if (!currentSessionId) return;
    
    try {
        const response = await fetch(`/api/progress/${currentSessionId}`);
        const data = await response.json();
        
        if (data.results) {
            console.log('Training results loaded:', data.results);
            // You can update the grid visualization here with the learned policy
        }
        
    } catch (error) {
        console.error('Error loading results:', error);
    }
}
// Add to gridworld.js

async function startTraining() {
    if (!currentAlgorithm) {
        showNotification('Please select an algorithm first!', 'warning');
        return;
    }
    
    if (isTraining) {
        showNotification('Training already in progress!', 'warning');
        return;
    }
    
    const params = getCurrentParameters();
    
    // Show what's being trained
    console.log(`Starting ${currentAlgorithm} with params:`, params);
    
    // Disable start button, enable stop button
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    
    // Reset chart
    trainingChart.data.labels = [];
    trainingChart.data.datasets[0].data = [];
    trainingChart.update();
    
    // Reset progress
    document.getElementById('training-progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    
    isTraining = true;
    document.getElementById('algorithm-status-text').textContent = 
        `${currentAlgorithm.replace('-', ' ').toUpperCase()} Training...`;
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                algorithm: currentAlgorithm,
                parameters: params
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        showNotification(`Training ${currentAlgorithm} started! Check console for progress.`, 'success');
        
        // Start progress monitoring
        startProgressMonitoring();
        
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Failed to start training: ' + error.message, 'error');
        stopTrainingUI();
    }
}
async function runInference() {
    if (!currentSessionId) {
        showNotification('Please train a model first', 'warning');
        return;
    }
    
    const startX = parseInt(document.getElementById('start-x').value) || 0;
    const startY = parseInt(document.getElementById('start-y').value) || 0;
    
    try {
        const response = await fetch('/api/run_inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                start_state: [startX, startY]
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showNotification(data.error, 'error');
            return;
        }
        
        // Update inference results
        const inferenceReward = document.getElementById('inference-reward');
        const inferenceSuccess = document.getElementById('inference-success');
        const inferenceSteps = document.getElementById('inference-steps');
        
        if (inferenceReward) inferenceReward.textContent = data.total_reward.toFixed(2);
        if (inferenceSuccess) inferenceSuccess.textContent = data.success ? 'Yes' : 'No';
        if (inferenceSteps) inferenceSteps.textContent = data.trajectory.length - 1;
        
        // Animate the trajectory
        animateTrajectory(data.trajectory);
        
        showNotification('Inference completed successfully!', 'success');
        
    } catch (error) {
        console.error('Error running inference:', error);
        showNotification('Error running inference', 'error');
    }
}

function animateTrajectory(trajectory) {
    if (!trajectory || trajectory.length === 0) return;
    
    let step = 0;
    const interval = setInterval(() => {
        if (step < trajectory.length) {
            // Update grid with current position
            // You would need to implement this based on your grid rendering
            console.log(`Step ${step}:`, trajectory[step]);
            step++;
        } else {
            clearInterval(interval);
        }
    }, 500);
}
// In gridworld.js, add this function:

function debugPolicy() {
    if (!trainingResults) {
        console.log("No training results available");
        return;
    }
    
    console.log("=== DEBUG POLICY ===");
    console.log("Algorithm:", trainingResults.algorithm);
    console.log("Policy matrix:");
    
    const policy = trainingResults.policy;
    const size = policy.length;
    
    // Create a readable policy display
    let policyDisplay = "";
    for (let i = 0; i < size; i++) {
        let row = "";
        for (let j = 0; j < size; j++) {
            const action = policy[i][j];
            let symbol;
            switch(action) {
                case 0: symbol = "↑"; break;
                case 1: symbol = "↓"; break;
                case 2: symbol = "←"; break;
                case 3: symbol = "→"; break;
                default: symbol = "?";
            }
            row += symbol + " ";
        }
        console.log(row);
        policyDisplay += row + "\n";
    }
    
    console.log("Value function:");
    const values = trainingResults.value_function;
    for (let i = 0; i < size; i++) {
        console.log(values[i].map(v => v.toFixed(2)).join(" "));
    }
    
    // Show in UI
    alert(`Policy Matrix:\n${policyDisplay}\nCheck console for detailed values.`);
}
// Add to gridworld.js

async function startTraining() {
    if (!currentAlgorithm) {
        showNotification('Please select an algorithm first!', 'warning');
        return;
    }
    
    if (isTraining) {
        showNotification('Training already in progress!', 'warning');
        return;
    }
    
    const params = getCurrentParameters();
    
    // Show what's being trained
    console.log(`Starting ${currentAlgorithm} with params:`, params);
    
    // Disable start button, enable stop button
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    
    // Reset chart
    trainingChart.data.labels = [];
    trainingChart.data.datasets[0].data = [];
    trainingChart.update();
    
    // Reset progress
    document.getElementById('training-progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    
    isTraining = true;
    document.getElementById('algorithm-status-text').textContent = 
        `${currentAlgorithm.replace('-', ' ').toUpperCase()} Training...`;
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                algorithm: currentAlgorithm,
                parameters: params
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        showNotification(`Training ${currentAlgorithm} started! Check console for progress.`, 'success');
        
        // Start progress monitoring
        startProgressMonitoring();
        
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Failed to start training: ' + error.message, 'error');
        stopTrainingUI();
    }
}

// gridworld.js - UPDATED PROGRESS MONITORING

let isTraining = false;
let currentAlgorithm = null;
let currentGridData = null;

function getCurrentParameters() {
    return {
        episodes: parseInt(document.getElementById('episodes').value) || 1000,
        gamma: parseFloat(document.getElementById('gamma').value) || 0.99,
        alpha: parseFloat(document.getElementById('alpha').value) || 0.1,
        epsilon: parseFloat(document.getElementById('epsilon').value) || 0.1
    };
}

async function startTraining() {
    if (!currentAlgorithm) {
        showNotification('Please select an algorithm first!', 'warning');
        return;
    }
    
    if (isTraining) {
        showNotification('Training already in progress!', 'warning');
        return;
    }
    
    const params = getCurrentParameters();
    
    // Update UI
    document.getElementById('start-training').disabled = true;
    document.getElementById('stop-training').disabled = false;
    isTraining = true;
    
    // Reset progress
    document.getElementById('training-progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithm: currentAlgorithm,
                parameters: params
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        showNotification(`Training ${currentAlgorithm} started!`, 'success');
        
        // Start monitoring progress
        monitorTrainingProgress();
        
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Failed to start training: ' + error.message, 'error');
        stopTrainingUI();
    }
}

async function monitorTrainingProgress() {
    let progressInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/progress');
            const status = await response.json();
            
            // Update progress
            const progress = status.progress || 0;
            document.getElementById('training-progress-bar').style.width = `${progress}%`;
            document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
            
            // Update episode count
            document.getElementById('current-episode').textContent = 
                `${status.current_episode || 0}/${status.total_episodes || 0}`;
            
            // Check if training is complete
            if (progress >= 100 || !status.is_training) {
                clearInterval(progressInterval);
                stopTrainingUI();
                
                if (progress >= 100) {
                    showNotification('Training completed successfully!', 'success');
                    document.getElementById('run-inference').disabled = false;
                }
            }
            
        } catch (error) {
            console.error('Error monitoring progress:', error);
            clearInterval(progressInterval);
            stopTrainingUI();
        }
    }, 1000); // Check every second
}

function stopTrainingUI() {
    document.getElementById('start-training').disabled = false;
    document.getElementById('stop-training').disabled = true;
    isTraining = false;
}
function resetTraining() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
    
    currentSessionId = null;
    
    // Reset UI
    const progressBar = document.getElementById('training-progress');
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
    }
    
    const currentEpisode = document.getElementById('current-episode');
    const currentReward = document.getElementById('current-reward');
    if (currentEpisode) currentEpisode.textContent = '0';
    if (currentReward) currentReward.textContent = '0.00';
    
    const startBtn = document.getElementById('start-training');
    const stopBtn = document.getElementById('stop-training');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
    
    // Clear chart
    if (trainingChart) {
        trainingChart.data.datasets[0].data = [];
        trainingChart.update();
    }
    
    showNotification('Training reset', 'info');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    notification.style.zIndex = '1060';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to notifications container or body
    const container = document.getElementById('notifications') || document.body;
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}
// Fixed inference function with proper error handling

async function runInference() {
    // Check if algorithm is trained
    if (!currentAlgorithm) {
        showNotification('No algorithm selected. Please select and train an algorithm first!', 'warning');
        return;
    }
    
    // Check if training is complete
    if (isTraining) {
        showNotification('Training is still in progress. Please wait for it to complete.', 'warning');
        return;
    }
    
    try {
        document.getElementById('training-status').innerHTML = 
            '<i class="fas fa-circle text-info me-2"></i>Running Inference...';
        
        const response = await fetch('/api/run_inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                start_state: gridData.start || [0, 0]
            })
        });
        
        const data = await response.json();
        
        // Handle case where no trained model exists
        if (!response.ok || data.error) {
            showNotification(
                data.message || 'No learned policy available. Please train an algorithm first!',
                'danger'
            );
            document.getElementById('training-status').innerHTML = 
                '<i class="fas fa-circle text-warning me-2"></i>No Policy Available';
            
            // Reset inference results
            document.getElementById('result-steps').textContent = '-';
            document.getElementById('result-reward').textContent = '-';
            document.getElementById('result-success').textContent = '-';
            return;
        }
        
        // Display results
        document.getElementById('result-steps').textContent = data.steps;
        document.getElementById('result-reward').textContent = data.total_reward.toFixed(2);
        document.getElementById('result-success').innerHTML = data.success ? 
            '<span class="text-success">✓ Yes</span>' : 
            '<span class="text-danger">✗ No</span>';
        
        // Show trajectory animation
        if (data.trajectory && data.trajectory.length > 0) {
            await showTrajectoryAnimated(data.trajectory);
        }
        
        document.getElementById('training-status').innerHTML = 
            '<i class="fas fa-circle text-success me-2"></i>Ready';
        
        const message = data.success ? 
            `Policy successfully reached goal in ${data.steps} steps!` :
            `Policy failed to reach goal. Try training with different parameters.`;
        
        showNotification(message, data.success ? 'success' : 'warning');
        
    } catch (error) {
        console.error('Error running inference:', error);
        showNotification('Failed to run inference: ' + error.message, 'danger');
        
        document.getElementById('training-status').innerHTML = 
            '<i class="fas fa-circle text-danger me-2"></i>Error';
    }
}

async function showTrajectoryAnimated(trajectory) {
    if (!trajectory || trajectory.length === 0 || !gridData) return;
    
    // Disable buttons during animation
    const btnInference = document.getElementById('btn-inference');
    const wasDisabled = btnInference.disabled;
    btnInference.disabled = true;
    
    // Reset grid to show agent at start
    gridData.agent_position = trajectory[0];
    renderGrid(gridData);
    
    // Animate the path with proper speed
    for (let step = 0; step < trajectory.length; step++) {
        gridData.agent_position = trajectory[step];
        renderGrid(gridData);
        
        // Wait between steps (faster animation)
        await new Promise(resolve => setTimeout(resolve, 300));
    }
    
    // Re-enable button
    btnInference.disabled = wasDisabled;
}

// Improved training status monitoring
function startProgressMonitoring() {
    if (progressInterval) clearInterval(progressInterval);
    
    let consecutiveErrors = 0;
    const maxErrors = 5;
    
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/training_status');
            const status = await response.json();
            
            // Reset error counter on success
            consecutiveErrors = 0;
            
            // Update progress bar
            const progress = Math.min(100, Math.max(0, status.progress || 0));
            const progressBar = document.getElementById('progress-bar');
            const progressText = document.getElementById('progress-text');
            
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `${Math.round(progress)}%`;
            
            // Update episode counters
            document.getElementById('current-episode').textContent = 
                status.current_episode || 0;
            document.getElementById('total-episodes').textContent = 
                status.total_episodes || 1000;
            
            // Update chart if we have new data
            if (status.history && status.history.rewards) {
                updateChart(status.history);
            }
            
            // Check if training is complete or stopped
            if (progress >= 100 || !status.is_training) {
                stopTraining();
                
                if (status.results && !status.results.error) {
                    showNotification('Training completed successfully!', 'success');
                    // Enable inference button
                    document.getElementById('btn-inference').disabled = false;
                } else if (status.results && status.results.error) {
                    showNotification('Training failed: ' + status.results.error, 'danger');
                }
            }
            
        } catch (error) {
            console.error('Error checking progress:', error);
            consecutiveErrors++;
            
            // Stop monitoring if too many errors
            if (consecutiveErrors >= maxErrors) {
                console.error('Too many consecutive errors, stopping monitoring');
                stopTraining();
                showNotification('Lost connection to training process', 'danger');
            }
        }
    }, 1000);
}

// Enhanced chart update with better visualization
function updateChart(history) {
    if (!history || !history.rewards || !history.episodes) return;
    
    trainingChart.data.labels = history.episodes;
    trainingChart.data.datasets[0].data = history.rewards;
    
    // Calculate moving average for smoother visualization
    const movingAvg = [];
    const windowSize = Math.min(10, Math.floor(history.rewards.length / 10) || 1);
    
    for (let i = 0; i < history.rewards.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const slice = history.rewards.slice(start, i + 1);
        const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
        movingAvg.push(avg);
    }
    
    trainingChart.data.datasets[1].data = movingAvg;
    trainingChart.update('none'); // Update without animation for smoother experience
    
    // Update training history
    trainingHistory = history.rewards;
    updateStats();
}

// Improved algorithm selection with validation
function selectAlgorithm(algorithm) {
    // Don't allow selection during training
    if (isTraining) {
        showNotification('Cannot change algorithm while training is in progress', 'warning');
        return;
    }
    
    // Deselect all
    document.querySelectorAll('.algorithm-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Select clicked algorithm
    const algoId = `algo-${algorithm.replace('-', '')}`;
    const algoCard = document.getElementById(algoId);
    if (algoCard) {
        algoCard.classList.add('selected');
    }
    
    currentAlgorithm = algorithm;
    document.getElementById('selected-algo').textContent = 
        algorithm.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    
    // Enable training button
    document.getElementById('btn-train').disabled = false;
    
    // Disable inference until training is complete
    document.getElementById('btn-inference').disabled = true;
    
    console.log(`Algorithm selected: ${algorithm}`);
    showNotification(`Selected algorithm: ${algorithm}`, 'info');
}

// Enhanced notification system with auto-dismiss
function showNotification(message, type = 'info') {
    // Remove existing notifications
    document.querySelectorAll('.notification-toast').forEach(el => el.remove());
    
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} alert-dismissible fade show position-fixed notification-toast`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);';
    
    // Icon based on type
    const icons = {
        'success': 'fa-check-circle',
        'danger': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };
    const icon = icons[type] || icons['info'];
    
    toast.innerHTML = `
        <i class="fas ${icon} me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

// Validate training parameters before starting
function validateTrainingParameters() {
    const params = getParameters();
    
    if (params.episodes < 100 || params.episodes > 10000) {
        showNotification('Episodes should be between 100 and 10000', 'warning');
        return false;
    }
    
    if (params.gamma < 0.5 || params.gamma > 1) {
        showNotification('Gamma should be between 0.5 and 1', 'warning');
        return false;
    }
    
    if (params.alpha < 0.01 || params.alpha > 1) {
        showNotification('Alpha should be between 0.01 and 1', 'warning');
        return false;
    }
    
    if (params.epsilon < 0.01 || params.epsilon > 0.5) {
        showNotification('Epsilon should be between 0.01 and 0.5', 'warning');
        return false;
    }
    
    return true;
}

// Enhanced start training with validation
async function startTraining() {
    if (!currentAlgorithm) {
        showNotification('Please select an algorithm first!', 'warning');
        return;
    }
    
    if (!validateTrainingParameters()) {
        return;
    }
    
    const params = getParameters();
    
    // Add confirmation for long training
    if (params.episodes > 2000) {
        if (!confirm(`Training with ${params.episodes} episodes may take a while. Continue?`)) {
            return;
        }
    }
    
    // Update UI
    isTraining = true;
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('btn-inference').disabled = true;
    document.getElementById('training-status').innerHTML = 
        '<i class="fas fa-spinner fa-spin me-2"></i>Training...';
    
    // Reset chart and results
    trainingChart.data.labels = [];
    trainingChart.data.datasets[0].data = [];
    trainingChart.data.datasets[1].data = [];
    trainingChart.update();
    trainingHistory = [];
    
    // Reset inference results
    document.getElementById('result-steps').textContent = '-';
    document.getElementById('result-reward').textContent = '-';
    document.getElementById('result-success').textContent = '-';
    
    // Start training
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                algorithm: currentAlgorithm,
                parameters: params
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        console.log('Training started:', data);
        showNotification(`Training ${currentAlgorithm} started with ${params.episodes} episodes`, 'info');
        
        // Start progress monitoring
        startProgressMonitoring();
        
    } catch (error) {
        console.error('Error starting training:', error);
        showNotification('Failed to start training: ' + error.message, 'danger');
        stopTraining();
    }
}
// Manual agent control functions
async function moveAgent(direction) {
    let action;
    switch(direction) {
        case 'up': action = 0; break;
        case 'down': action = 1; break;
        case 'left': action = 2; break;
        case 'right': action = 3; break;
        default: return;
    }
    
    try {
        const response = await fetch('/api/move_agent', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ action: action })
        });
        
        const data = await response.json();
        
        // Reload grid to show new position
        initGrid();
        
        if (data.reached_goal) {
            showNotification('🎉 Goal reached! 🎉', 'success');
        }
        
    } catch (error) {
        console.error('Error moving agent:', error);
        showNotification('Failed to move agent', 'error');
    }
}