import unittest
import torch
import numpy as np
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent
from train import PPOTrainer


class TestTrainingIntegration(unittest.TestCase):
    """Integration test for training with LSTM hidden state fix"""
    
    def test_training_runs_without_error(self):
        """Test that training runs without LSTM hidden state mismatch errors"""
        # Create environment
        env = SnakeGame(grid_size=10, cell_size=30)
        
        # Create agent
        input_shape = (env.grid_size, env.grid_size, 3)
        action_size = 4
        agent = PPO_LSTM_Agent(input_shape, action_size)
        
        # Create trainer (without early stopping for this test)
        trainer = PPOTrainer(env, agent, batch_size=64, early_stopping_patience=None)
        
        # Run a few iterations - should not raise RuntimeError
        try:
            trainer.train(num_iterations=2, episodes_per_iteration=2, save_interval=100)
            success = True
        except RuntimeError as e:
            if "Expected hidden" in str(e):
                success = False
            else:
                raise
        finally:
            env.close()
        
        self.assertTrue(success, "Training failed with LSTM hidden state mismatch")
    
    def test_collect_and_update_cycle(self):
        """Test that collect and update cycle works with different batch sizes"""
        env = SnakeGame(grid_size=10, cell_size=30)
        input_shape = (env.grid_size, env.grid_size, 3)
        action_size = 4
        agent = PPO_LSTM_Agent(input_shape, action_size)
        
        # Test with various batch sizes
        for batch_size in [16, 32, 64]:
            with self.subTest(batch_size=batch_size):
                trainer = PPOTrainer(env, agent, batch_size=batch_size, 
                                   early_stopping_patience=None)
                
                # Collect experiences (action selection uses batch_size=1)
                episode_rewards = trainer.collect_experiences(num_episodes=2)
                self.assertIsInstance(episode_rewards, list)
                
                # Update (uses the specified batch_size)
                loss = trainer.update()
                self.assertIsInstance(loss, (int, float))
        
        env.close()


if __name__ == '__main__':
    unittest.main()
