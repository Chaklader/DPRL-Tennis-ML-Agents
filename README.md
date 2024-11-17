# Multi-Agent Deep Reinforcement Learning for Tennis

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

## Project Description

This project implements a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train two agents to play tennis in a Unity ML-Agents environment. The agents must learn to cooperatively bounce a ball over a net, making it a challenging multi-agent continuous control task.

### Environment Details

#### State Space (Per Agent)
- 8 variables representing:
  - Position of ball and racket
  - Velocity of ball and racket
- Each agent receives its own local observation

#### Action Space
- 2 continuous actions:
  - Movement toward/away from net
  - Jumping
- Actions are bounded in [-1, 1]

#### Reward Structure
- +0.1 for hitting ball over net
- -0.01 for:
  - Letting ball hit ground
  - Hitting ball out of bounds

#### Success Criteria
- Average score of +0.5 over 100 consecutive episodes
- Score calculation:
  1. Sum rewards for each agent per episode
  2. Take maximum of both agents' scores
  3. Average these maxima over 100 episodes

## Installation

### 1. Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate udacity_rl_new
```


### 2. Download Tennis Environment

Choose your OS:
```bash
# Linux (64-bit)
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
unzip Tennis_Linux.zip

# For AWS (headless)
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip
```


Other platforms:
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Running the Project

### 1. Training
```bash
jupyter notebook Tennis.ipynb
```


Follow notebook cells for:
- Environment initialization
- Agent configuration
- Training execution
- Performance visualization

### 2. Configuration Options
- Network architecture:
  - Actor: 2 hidden layers [64, 64]
  - Critic: 2 hidden layers [64, 64]
- Learning rates:
  - Actor: 3e-3
  - Critic: 4e-4
- Replay buffer size: 1e6
- Batch size: 64
- Discount factor (γ): 0.99
- Soft update factor (τ): 8e-3

### 3. Training Process
- Episodes run until environment solved
- Learning rate scheduling at checkpoints
- Progress tracking:
  - Episode scores
  - Moving averages
  - Agent performance metrics

### 4. Testing
- Load trained model weights
- Disable exploration noise
- Visualize agent performance
- Analyze training results


---