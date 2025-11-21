import pygame
import torch
import numpy as np
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent
import argparse

def test_agent(model_path, grid_size=10, cell_size=30, num_episodes=5, render_delay=100):
    """
    Test a trained PPO LSTM agent on the Snake game
    
    Args:
        model_path: path to the trained model
        grid_size: size of the game grid
        cell_size: size of each cell in pixels
        num_episodes: number of episodes to test
        render_delay: delay between frames in milliseconds
    """
    # Create environment
    env = SnakeGame(grid_size=grid_size, cell_size=cell_size)
    
    # Create agent
    input_shape = (env.grid_size, env.grid_size, 3)
    action_size = 4
    agent = PPO_LSTM_Agent(input_shape, action_size)
    
    # Load trained model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test the agent
    total_rewards = []
    total_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_hidden_state()
        
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            # Get action from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, _ = agent.forward(state_tensor, reset_hidden=(steps == 0))
                
            # Select best action (no exploration during testing)
            action = torch.argmax(action_probs, dim=-1).item()
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Render
            env.render()
            pygame.time.delay(render_delay)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_scores.append(info['score'])
        print(f"Episode {episode + 1} completed!")
        print(f"Score: {info['score']}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Steps: {steps}")
        
        # Wait a bit between episodes
        pygame.time.delay(1000)
    
    # Print statistics
    print(f"\n{'='*50}")
    print("Testing Results:")
    print(f"{'='*50}")
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Max Score: {np.max(total_scores)}")
    print(f"Min Score: {np.min(total_scores)}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    
    env.close()

def interactive_test(model_path, grid_size=10, cell_size=30):
    """
    Interactive test where you can control the game speed and pause
    
    Args:
        model_path: path to the trained model
        grid_size: size of the game grid
        cell_size: size of each cell in pixels
    """
    # Create environment
    env = SnakeGame(grid_size=grid_size, cell_size=cell_size)
    
    # Create agent
    input_shape = (env.grid_size, env.grid_size, 3)
    action_size = 4
    agent = PPO_LSTM_Agent(input_shape, action_size)
    
    # Load trained model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nInteractive Controls:")
    print("- SPACE: Pause/Resume")
    print("- UP/DOWN: Adjust speed")
    print("- R: Restart episode")
    print("- ESC: Exit")
    print("\nStarting interactive test...")
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    speed = 10  # FPS
    
    while running:
        state = env.reset()
        agent.reset_hidden_state()
        
        episode_done = False
        
        while not episode_done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        speed = min(60, speed + 5)
                        print(f"Speed: {speed} FPS")
                    elif event.key == pygame.K_DOWN:
                        speed = max(1, speed - 5)
                        print(f"Speed: {speed} FPS")
                    elif event.key == pygame.K_r:
                        episode_done = True
                        state = env.reset()
                        agent.reset_hidden_state()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        episode_done = True
            
            if not paused and not episode_done:
                # Get action from agent
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs, _ = agent.forward(state_tensor, reset_hidden=True)
                    
                # Select best action
                action = torch.argmax(action_probs, dim=-1).item()
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                if done:
                    episode_done = True
                    print(f"Episode finished! Score: {info['score']}")
                
                state = next_state
            
            # Render
            env.render()
            clock.tick(speed)
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Test PPO LSTM Snake Agent')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to test')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='Size of the game grid')
    parser.add_argument('--cell_size', type=int, default=30,
                       help='Size of each cell in pixels')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in milliseconds')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test(args.model, args.grid_size, args.cell_size)
    else:
        test_agent(args.model, args.grid_size, args.cell_size, 
                  args.episodes, args.delay)

if __name__ == "__main__":
    main()