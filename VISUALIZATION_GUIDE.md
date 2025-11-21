# Visualization Guide

## Overview

The `visualize.py` script provides real-time visualization of trained Snake game agents. It was developed using Test-Driven Development (TDD) methodology.

## Features

- **Auto-Detection**: Automatically finds the best trained model in the directory
- **Customizable Speed**: Adjust playback speed with FPS parameter
- **Statistics**: Shows detailed statistics after each episode and summary at the end
- **User Control**: Exit anytime with ESC key or window close
- **Error Handling**: Clear error messages when models are not found

## Quick Start

```bash
# Train a model first
python train.py

# Visualize with default settings
python visualize.py

# Custom settings
python visualize.py --fps 20 --episodes 10
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to specific model file | Auto-detect |
| `--grid_size` | Size of game grid | 10 |
| `--cell_size` | Size of each cell in pixels | 30 |
| `--episodes` | Number of episodes to run | 5 |
| `--fps` | Frames per second (speed) | 10 |

## Model Detection Priority

The script searches for models in this order:

1. `ppo_snake_agent_final.pth` (final trained model)
2. `ppo_snake_agent_interrupted.pth` (interrupted training)
3. Highest iteration checkpoint (`ppo_snake_agent_<N>.pth`)

## Usage Examples

### Basic Visualization
```bash
python visualize.py
```

### Fast Playback
```bash
python visualize.py --fps 30
```

### Slow Motion Analysis
```bash
python visualize.py --fps 5
```

### Multiple Episodes
```bash
python visualize.py --episodes 20
```

### Larger Grid
```bash
python visualize.py --grid_size 15 --fps 15
```

### Specific Model
```bash
python visualize.py --model my_best_model.pth
```

## Output

The visualization displays:

- **During Play**: Real-time game rendering with score
- **After Each Episode**:
  - Episode number
  - Final score
  - Total reward
  - Number of steps

- **Final Summary**:
  - Average/Max/Min scores
  - Average/Max/Min rewards
  - Average/Max/Min steps

## Controls

- **ESC**: Exit visualization
- **Close Window**: Exit visualization

## Testing

The visualization script includes comprehensive unit tests:

```bash
# Run all visualization tests
python -m unittest test_visualize -v

# Run specific test class
python -m unittest test_visualize.TestVisualizeAgent -v
```

### Test Coverage

- Model finding and loading
- Command-line argument parsing
- Episode execution
- Event handling (quit, ESC)
- FPS control
- Statistics display

All 16 tests pass successfully.

## TDD Development Process

This feature was developed using Test-Driven Development:

1. **Red Phase**: Wrote failing tests first
   - Created test cases for all functionality
   - Tests failed because implementation didn't exist

2. **Green Phase**: Implemented functionality
   - Wrote code to make all tests pass
   - Ensured proper error handling

3. **Refactor Phase**: Improved code quality
   - Added documentation
   - Improved error messages
   - Ensured consistent code style

## Troubleshooting

### "No trained model found"
- Train a model first: `python train.py`
- Or specify a model: `python visualize.py --model <path>`

### Slow Performance
- Increase FPS: `python visualize.py --fps 30`
- Decrease grid size: `python visualize.py --grid_size 8`

### Display Issues
- Ensure pygame is installed: `pip install pygame`
- Check display settings if running remotely

## Implementation Details

- **Agent Policy**: Uses greedy policy (argmax) for best action selection
- **Hidden State**: Resets LSTM hidden state at episode start
- **Rendering**: Uses pygame for real-time display
- **Error Handling**: Comprehensive try-except blocks with user-friendly messages

## Code Structure

```
visualize.py
├── find_best_model()      # Auto-detect best trained model
├── load_best_model()      # Load model checkpoint
├── visualize_agent()      # Main visualization loop
├── parse_args()           # Command-line argument parsing
└── main()                 # Entry point
```

## Dependencies

- torch
- numpy
- pygame
- argparse (standard library)
- os, glob, time (standard library)

## Related Files

- `snake_game.py`: Game environment
- `ppo_agent.py`: Agent neural network
- `train.py`: Training script
- `test_visualize.py`: Unit tests
