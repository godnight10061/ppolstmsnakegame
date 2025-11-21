import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import deque
import random
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent
import matplotlib.pyplot as plt


class TimeBasedEarlyStopping:
    def __init__(self, patience_seconds=300):
        if patience_seconds is None:
            raise ValueError("patience_seconds must be provided for early stopping")
        if patience_seconds <= 0:
            raise ValueError("patience_seconds must be a positive number")
        self.patience_seconds = patience_seconds
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.best_reward = None
        self.best_reward_time = None

    def update(self, reward):
        if reward is None:
            return False

        current_time = time.time()

        if self.best_reward is None or reward >= self.best_reward:
            self.best_reward = reward
            self.best_reward_time = current_time
            return False

        if self.best_reward_time is None:
            self.best_reward_time = current_time
            return False

        elapsed = current_time - self.best_reward_time
        return elapsed >= self.patience_seconds

    def time_since_improvement(self):
        reference_time = self.best_reward_time or self.start_time
        return time.time() - reference_time

    def get_state_dict(self):
        return {
            'patience_seconds': self.patience_seconds,
            'start_time': self.start_time,
            'best_reward': self.best_reward,
            'best_reward_time': self.best_reward_time
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            self.reset()
            return

        self.patience_seconds = state_dict.get('patience_seconds', self.patience_seconds)
        self.start_time = state_dict.get('start_time', time.time())
        self.best_reward = state_dict.get('best_reward')
        self.best_reward_time = state_dict.get('best_reward_time')


class PPOTrainer:
    def __init__(self, env, agent, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epoch=4, 
                 batch_size=64, memory_size=2048, early_stopping_patience=300):
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
            early_stopping_patience: patience in seconds for early stopping (None to disable)
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
        
        # Early stopping
        self.early_stopping = None
        if early_stopping_patience is not None:
            self.early_stopping = TimeBasedEarlyStopping(patience_seconds=early_stopping_patience)
        
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
    
    def train(self, num_iterations=None, episodes_per_iteration=10, save_interval=100):
        """Main training loop"""
        print("Starting training...")
        iteration = 0
        
        while num_iterations is None or iteration < num_iterations:
            # Collect experiences
            episode_rewards = self.collect_experiences(episodes_per_iteration)
            self.episode_rewards.extend(episode_rewards)
            
            # Update agent
            loss = self.update()
            self.losses.append(loss)
            
            avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
            
            # Save model
            if iteration % save_interval == 0 and iteration > 0:
                self.save_model(f"ppo_snake_agent_{iteration}.pth")
                print(f"Model saved at iteration {iteration}")
            
            # Early stopping check
            if self.early_stopping and self.early_stopping.update(avg_reward):
                wait_time = self.early_stopping.time_since_improvement()
                print(
                    f"Early stopping triggered after {wait_time:.2f} seconds without improvement."
                )
                break
            
            iteration += 1
        
        print("Training completed!")
    
    def save_model(self, filepath):
        """Save the trained model"""
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
        if self.early_stopping:
            checkpoint['early_stopping_state'] = self.early_stopping.get_state_dict()
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        early_state = checkpoint.get('early_stopping_state')
        if early_state:
            if self.early_stopping is None:
                self.early_stopping = TimeBasedEarlyStopping(
                    patience_seconds=early_state.get('patience_seconds', 300)
                )
            self.early_stopping.load_state_dict(early_state)
    
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
    
    # Create trainer with 5 minutes early stopping patience
    trainer = PPOTrainer(env, agent, early_stopping_patience=300)  # 5 minutes = 300 seconds
    
    # Train the agent with infinite iterations (will stop on early stopping)
    try:
        print("Starting infinite training with 5 minute early stopping patience...")
        trainer.train(num_iterations=None, episodes_per_iteration=5, save_interval=100)
        
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