# PPO LSTM Snake Game Agent

A reinforcement learning implementation of a Snake game agent using Proximal Policy Optimization (PPO) with LSTM memory for sequential decision making.

## Features

- **Snake Game Environment**: Custom implementation using Pygame
- **PPO Algorithm**: State-of-the-art reinforcement learning algorithm
- **LSTM Memory**: Enables the agent to remember past states and actions
- **CNN Feature Extraction**: Processes visual game state efficiently
- **Training and Testing Scripts**: Complete pipeline for training and evaluation

## Project Structure

```
├── snake_game.py      # Snake game environment
├── ppo_agent.py       # PPO LSTM neural network agent
├── train.py           # Training script
├── test_agent.py      # Testing and evaluation script
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agent from scratch:

```bash
python train.py
```

The training script will:
- Create a Snake game environment (10x10 grid)
- Initialize the PPO LSTM agent
- Train with infinite iterations (controlled by early stopping)
- Use time-based early stopping with 5 minutes patience
- Stop automatically when no reward improvement for 5 minutes
- Save model checkpoints every 100 iterations
- Plot training progress

The training will run indefinitely until:
1. Early stopping is triggered (no improvement for 5 minutes)
2. User interrupts with Ctrl+C

### Testing

To test a trained agent:

```bash
# Basic testing (5 episodes)
python test_agent.py --model ppo_snake_agent_final.pth

# Custom testing parameters
python test_agent.py --model ppo_snake_agent_final.pth --episodes 10 --grid_size 15

# Interactive mode with controls
python test_agent.py --model ppo_snake_agent_final.pth --interactive
```

Interactive controls:
- **SPACE**: Pause/Resume
- **UP/DOWN**: Adjust speed
- **R**: Restart episode
- **ESC**: Exit

## Architecture

### Neural Network

The PPO LSTM agent uses a hybrid architecture:

1. **CNN Layers**: Extract spatial features from the game grid
   - 3 convolutional layers with max pooling
   - Processes 3-channel input (snake body, head, food)

2. **Fully Connected Layers**: Bridge CNN features to LSTM
   - Two dense layers for feature transformation

3. **LSTM Layer**: Provides memory for sequential decision making
   - Remembers past states and actions
   - Handles temporal dependencies in the game

4. **Actor-Critic Heads**:
   - **Actor**: Outputs action probabilities (4 directions)
   - **Critic**: Estimates state value for advantage calculation

### PPO Algorithm

The implementation uses:
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimates
- **Mini-batch Updates**: Efficient memory usage and training
- **Entropy Regularization**: Encourages exploration

## Game State Representation

The game state is represented as a 3-channel grid:
- **Channel 0**: Snake body positions
- **Channel 1**: Snake head position
- **Channel 2**: Food position

## Reward Structure

- **+10**: Eating food
- **-10**: Hitting wall or self
- **-5**: Too many steps without food
- **-0.01**: Each step (encourages efficiency)

## Hyperparameters

Key hyperparameters used:
- **Learning Rate**: 3e-4
- **Discount Factor (γ)**: 0.99
- **Clipping Parameter (ε)**: 0.2
- **LSTM Hidden Size**: 128
- **CNN Hidden Size**: 256
- **Batch Size**: 64
- **Update Epochs**: 4

## Performance

The agent typically learns to:
- Navigate towards food efficiently
- Avoid walls and self-collision
- Maintain longer episodes over time
- Achieve scores of 10+ on a 10x10 grid after training

## Customization

You can modify various parameters:
- Grid size in `SnakeGame(grid_size=N)`
- Network architecture in `PPO_LSTM_Agent`
- Training hyperparameters in `PPOTrainer`
- Reward structure in `SnakeGame.step()`

## Troubleshooting

1. **Training is slow**: Reduce grid size or batch size
2. **Agent not learning**: Check reward structure and hyperparameters
3. **Memory issues**: Reduce memory buffer size or batch size
4. **Display issues**: Ensure pygame is properly installed

## Future Improvements

- Curriculum learning with increasing grid sizes
- Attention mechanisms for better spatial reasoning
- Multi-agent training (snake vs snake)
- Transfer learning to similar grid-based games