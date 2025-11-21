import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent
import matplotlib.pyplot as plt

class PPOTrainer:
    def __init__(self, env, agent, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epoch=4, 
                 batch_size=64, memory_size=2048):
        """
        PPO Trainer with LSTM memory
        
        Args:
            env: environment (SnakeGame)
            agent: PPO LSTM agent
            lr: learning rate
            gamma: discount factor
            eps_clip: PPO clipping parameter
            k_epoch: number of epochs for each update
            batch_size: mini-batch size
            memory_size: size of experience buffer
        """
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epoch = k_epoch
        self.batch_size = batch_size
        
        # Memory for storing experiences
        self.memory = []
        self.memory_size = memory_size
        
        # Optimizer
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        
        # Loss coefficients
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Training statistics
        self.episode_rewards = []
        self.losses = []
        
    def collect_experiences(self, num_episodes=10):
        """Collect experiences by running episodes"""
        self.memory = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            self.agent.reset_hidden_state()
            
            episode_reward = 0
            done = False
            episode_data = []
            
            while not done:
                # Select action
                action, log_prob, value = self.agent.select_action(state)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                episode_data.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'log_prob': log_prob,
                    'value': value,
                    'done': done
                })
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Calculate advantages and returns for this episode
            advantages, returns = self._calculate_advantages_returns(episode_data)
            
            # Add to memory
            for i, data in enumerate(episode_data):
                data['advantage'] = advantages[i]
                data['return'] = returns[i]
                self.memory.append(data)
        
        # Limit memory size
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
        
        return episode_rewards
    
    def _calculate_advantages_returns(self, episode_data):
        """Calculate advantages and returns using GAE"""
        advantages = []
        returns = []
        
        # Calculate returns (discounted rewards)
        R = 0
        for data in reversed(episode_data):
            R = data['reward'] + (self.gamma * R * (1 - data['done']))
            returns.insert(0, R)
        
        # Calculate advantages (return - value)
        for i, data in enumerate(episode_data):
            advantage = returns[i] - data['value']
            advantages.append(advantage)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages.tolist(), returns
    
    def update(self):
        """Update the agent using PPO algorithm"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Convert memory to tensors
        states = torch.FloatTensor([data['state'] for data in self.memory])
        actions = torch.LongTensor([data['action'] for data in self.memory])
        old_log_probs = torch.FloatTensor([data['log_prob'] for data in self.memory])
        advantages = torch.FloatTensor([data['advantage'] for data in self.memory])
        returns = torch.FloatTensor([data['return'] for data in self.memory])
        
        total_loss = 0
        
        # PPO update
        for _ in range(self.k_epoch):
            # Shuffle indices
            indices = torch.randperm(len(self.memory))
            
            # Mini-batch updates
            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                new_log_probs, values, entropy = self.agent.evaluate_actions(
                    batch_states, batch_actions)
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clipped loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / self.k_epoch
    
    def train(self, num_iterations=1000, episodes_per_iteration=10, save_interval=100):
        """Main training loop"""
        print("Starting training...")
        
        for iteration in range(num_iterations):
            # Collect experiences
            episode_rewards = self.collect_experiences(episodes_per_iteration)
            self.episode_rewards.extend(episode_rewards)
            
            # Update agent
            loss = self.update()
            self.losses.append(loss)
            
            # Print progress
            if iteration % 10 == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
            
            # Save model
            if iteration % save_interval == 0 and iteration > 0:
                self.save_model(f"ppo_snake_agent_{iteration}.pth")
                print(f"Model saved at iteration {iteration}")
        
        print("Training completed!")
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot episode rewards
        if self.episode_rewards:
            # Calculate moving average
            window_size = 100
            moving_avg = []
            for i in range(len(self.episode_rewards)):
                start = max(0, i - window_size + 1)
                moving_avg.append(np.mean(self.episode_rewards[start:i+1]))
            
            ax1.plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
            ax1.plot(moving_avg, label=f'Moving Avg ({window_size} episodes)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Progress - Rewards')
            ax1.legend()
            ax1.grid(True)
        
        # Plot losses
        if self.losses:
            ax2.plot(self.losses)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Progress - Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Create environment
    env = SnakeGame(grid_size=10, cell_size=30)
    
    # Create agent
    input_shape = (env.grid_size, env.grid_size, 3)  # height, width, channels
    action_size = 4  # up, right, down, left
    agent = PPO_LSTM_Agent(input_shape, action_size)
    
    # Create trainer
    trainer = PPOTrainer(env, agent)
    
    # Train the agent
    try:
        trainer.train(num_iterations=500, episodes_per_iteration=5, save_interval=100)
        
        # Plot progress
        trainer.plot_training_progress()
        
        # Save final model
        trainer.save_model("ppo_snake_agent_final.pth")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        trainer.save_model("ppo_snake_agent_interrupted.pth")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()