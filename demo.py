#!/usr/bin/env python3
"""
Quick demo script to test the PPO LSTM Snake agent with minimal training.
This script runs a short training session and then demonstrates the agent.
"""

import torch
import numpy as np
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent
from train import PPOTrainer

def quick_demo():
    """Quick demonstration with minimal training"""
    print("=== PPO LSTM Snake Agent Quick Demo ===\n")
    
    # Create environment
    print("Creating Snake game environment...")
    env = SnakeGame(grid_size=8, cell_size=40)  # Smaller grid for faster training
    
    # Create agent
    input_shape = (env.grid_size, env.grid_size, 3)
    action_size = 4
    agent = PPO_LSTM_Agent(input_shape, action_size, hidden_size=128, lstm_hidden_size=64)
    
    # Create trainer with reduced parameters for quick demo
    trainer = PPOTrainer(env, agent, lr=1e-3, batch_size=32, memory_size=512)
    
    print("Starting quick training (50 iterations)...")
    print("This will take a few minutes...\n")
    
    try:
        # Quick training
        trainer.train(num_iterations=50, episodes_per_iteration=3, save_interval=25)
        
        print("\nTraining completed!")
        print("Testing the trained agent...\n")
        
        # Test the agent
        test_episodes = 3
        total_scores = []
        
        for episode in range(test_episodes):
            state = env.reset()
            agent.reset_hidden_state()
            
            episode_reward = 0
            done = False
            steps = 0
            
            print(f"Test Episode {episode + 1}:")
            
            while not done and steps < 200:  # Limit steps for demo
                # Get action from agent
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs, _ = agent.forward(state_tensor, reset_hidden=(steps == 0))
                
                # Select best action
                action = torch.argmax(action_probs, dim=-1).item()
                
                # Take step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Render every few steps
                if steps % 5 == 0:
                    env.render()
                
                state = next_state
            
            total_scores.append(info['score'])
            print(f"  Score: {info['score']}, Steps: {steps}, Reward: {episode_reward:.2f}")
        
        print(f"\nDemo Results:")
        print(f"Average Score: {np.mean(total_scores):.2f}")
        print(f"Best Score: {np.max(total_scores)}")
        
        # Save the demo model
        trainer.save_model("demo_ppo_snake_agent.pth")
        print("\nDemo model saved as 'demo_ppo_snake_agent.pth'")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    finally:
        env.close()

def random_baseline_demo():
    """Show random agent performance for comparison"""
    print("\n=== Random Agent Baseline ===\n")
    
    env = SnakeGame(grid_size=8, cell_size=40)
    test_episodes = 3
    total_scores = []
    
    print("Testing random agent for comparison...")
    
    for episode in range(test_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            # Random action
            action = np.random.randint(0, 4)
            next_state, reward, done, info = env.step(action)
            steps += 1
            
            # Render every few steps
            if steps % 5 == 0:
                env.render()
            
            state = next_state
        
        total_scores.append(info['score'])
        print(f"Random Episode {episode + 1}: Score {info['score']}, Steps {steps}")
    
    print(f"\nRandom Agent Results:")
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Best Score: {np.max(total_scores)}")
    
    env.close()

if __name__ == "__main__":
    try:
        # First show random baseline
        random_baseline_demo()
        
        # Then train and test PPO agent
        quick_demo()
        
        print("\n=== Demo Complete ===")
        print("You can now:")
        print("1. Run 'python train.py' for full training")
        print("2. Run 'python test_agent.py --model demo_ppo_snake_agent.pth' to test the demo model")
        print("3. Run 'python test_agent.py --model demo_ppo_snake_agent.pth --interactive' for interactive testing")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")