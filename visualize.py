#!/usr/bin/env python3
"""
Visualize the best trained agent playing Snake in real-time

This script loads the best trained model and visualizes the agent playing
the Snake game in real-time with customizable speed and display options.
"""

import pygame
import torch
import numpy as np
import os
import glob
import argparse
import time
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent


def find_best_model(search_dir="."):
    """
    Find the best trained model in the given directory
    
    Priority:
    1. ppo_snake_agent_final.pth (final trained model)
    2. Highest iteration checkpoint (ppo_snake_agent_<iteration>.pth)
    
    Args:
        search_dir: directory to search for models
        
    Returns:
        path to the best model
        
    Raises:
        FileNotFoundError: if no trained model is found
    """
    # Check for final model
    final_model = os.path.join(search_dir, "ppo_snake_agent_final.pth")
    if os.path.exists(final_model):
        return final_model
    
    # Check for interrupted model
    interrupted_model = os.path.join(search_dir, "ppo_snake_agent_interrupted.pth")
    if os.path.exists(interrupted_model):
        return interrupted_model
    
    # Find checkpoint models
    pattern = os.path.join(search_dir, "ppo_snake_agent_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No trained model found in {search_dir}. "
            "Please train a model first using train.py"
        )
    
    # Extract iteration numbers and find highest
    checkpoints = []
    for filepath in checkpoint_files:
        basename = os.path.basename(filepath)
        # Extract number from ppo_snake_agent_<number>.pth
        try:
            iteration = int(basename.replace("ppo_snake_agent_", "").replace(".pth", ""))
            checkpoints.append((iteration, filepath))
        except ValueError:
            continue
    
    if not checkpoints:
        raise FileNotFoundError(
            f"No valid checkpoint found in {search_dir}. "
            "Please train a model first using train.py"
        )
    
    # Return checkpoint with highest iteration
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def load_best_model(model_path=None, search_dir=".", grid_size=10):
    """
    Load the best trained model
    
    Args:
        model_path: explicit path to model file (if None, will search)
        search_dir: directory to search for models if model_path is None
        grid_size: size of the game grid
        
    Returns:
        loaded PPO_LSTM_Agent
        
    Raises:
        FileNotFoundError: if model not found
        Exception: if model loading fails
    """
    # Find model if not explicitly provided
    if model_path is None:
        model_path = find_best_model(search_dir)
    
    print(f"Loading model from: {model_path}")
    
    # Create agent
    input_shape = (grid_size, grid_size, 3)
    action_size = 4
    agent = PPO_LSTM_Agent(input_shape, action_size)
    
    # Load model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        return agent
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")


def visualize_agent(agent, grid_size=10, cell_size=30, num_episodes=5, fps=10):
    """
    Visualize the agent playing Snake in real-time
    
    Args:
        agent: trained PPO_LSTM_Agent
        grid_size: size of the game grid
        cell_size: size of each cell in pixels
        num_episodes: number of episodes to visualize
        fps: frames per second (controls visualization speed)
    """
    # Create environment
    env = SnakeGame(grid_size=grid_size, cell_size=cell_size)
    
    # Statistics tracking
    total_scores = []
    total_rewards = []
    total_steps = []
    
    print("\n" + "="*60)
    print("STARTING VISUALIZATION")
    print("="*60)
    print(f"Episodes: {num_episodes}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"FPS: {fps}")
    print("\nPress ESC or close window to exit")
    print("="*60 + "\n")
    
    clock = pygame.time.Clock()
    
    for episode in range(num_episodes):
        # Reset environment and agent
        state = env.reset()
        agent.reset_hidden_state()
        
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nVisualization interrupted by user")
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("\nVisualization interrupted by user")
                        env.close()
                        return
            
            # Get action from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, value = agent.forward(state_tensor, reset_hidden=(steps == 0))
            
            # Select best action (greedy policy for visualization)
            action = torch.argmax(action_probs, dim=-1).item()
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update statistics
            episode_reward += reward
            steps += 1
            
            # Render the game
            env.render()
            
            # Control frame rate
            clock.tick(fps)
            
            state = next_state
        
        # Episode completed
        score = info.get('score', 0)
        total_scores.append(score)
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        print(f"Score: {score}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Steps: {steps}")
        
        # Pause briefly between episodes
        time.sleep(1)
    
    # Print final statistics
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Episodes Completed: {num_episodes}")
    print(f"\nScore Statistics:")
    print(f"  Average: {np.mean(total_scores):.2f}")
    print(f"  Max: {np.max(total_scores)}")
    print(f"  Min: {np.min(total_scores)}")
    print(f"\nReward Statistics:")
    print(f"  Average: {np.mean(total_rewards):.2f}")
    print(f"  Max: {np.max(total_rewards):.2f}")
    print(f"  Min: {np.min(total_rewards):.2f}")
    print(f"\nSteps Statistics:")
    print(f"  Average: {np.mean(total_steps):.1f}")
    print(f"  Max: {np.max(total_steps)}")
    print(f"  Min: {np.min(total_steps)}")
    print("="*60)
    
    env.close()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize best trained PPO Snake agent in real-time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find and visualize best model in current directory
  python visualize.py
  
  # Specify a model file
  python visualize.py --model ppo_snake_agent_final.pth
  
  # Custom visualization settings
  python visualize.py --grid_size 15 --fps 20 --episodes 10
  
  # Fast playback
  python visualize.py --fps 30
  
  # Slow playback for analysis
  python visualize.py --fps 5
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model file (auto-detects if not specified)'
    )
    
    parser.add_argument(
        '--grid_size',
        type=int,
        default=10,
        help='Size of the game grid (default: 10)'
    )
    
    parser.add_argument(
        '--cell_size',
        type=int,
        default=30,
        help='Size of each cell in pixels (default: 30)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to visualize (default: 5)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second - controls playback speed (default: 10)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for visualization"""
    args = parse_args()
    
    try:
        # Load the best model
        agent = load_best_model(
            model_path=args.model,
            search_dir=".",
            grid_size=args.grid_size
        )
        
        # Visualize the agent
        visualize_agent(
            agent=agent,
            grid_size=args.grid_size,
            cell_size=args.cell_size,
            num_episodes=args.episodes,
            fps=args.fps
        )
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train a model first using:")
        print("  python train.py")
        exit(1)
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
        exit(0)
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
