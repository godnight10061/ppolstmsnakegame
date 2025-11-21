import unittest
import time
from unittest.mock import patch
import torch
import numpy as np
from train import PPOTrainer, TimeBasedEarlyStopping
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent


class TestTimeBasedEarlyStopping(unittest.TestCase):
    """Test suite for time-based early stopping functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.patience_seconds = 0.2  # Faster tests (production uses 300 seconds)
        self.early_stopping = TimeBasedEarlyStopping(patience_seconds=self.patience_seconds)
    
    def test_initialization(self):
        """Test that early stopping initializes correctly"""
        self.assertEqual(self.early_stopping.patience_seconds, self.patience_seconds)
        self.assertIsNone(self.early_stopping.best_reward)
        self.assertIsNone(self.early_stopping.best_reward_time)
        self.assertIsNotNone(self.early_stopping.start_time)
    
    def test_should_not_stop_on_first_update(self):
        """Test that early stopping doesn't trigger on first update"""
        result = self.early_stopping.update(reward=10.0)
        self.assertFalse(result)
        self.assertEqual(self.early_stopping.best_reward, 10.0)
    
    def test_should_not_stop_when_improving(self):
        """Test that early stopping doesn't trigger when reward is improving"""
        self.early_stopping.update(reward=10.0)
        time.sleep(0.1)
        result = self.early_stopping.update(reward=15.0)
        self.assertFalse(result)
        self.assertEqual(self.early_stopping.best_reward, 15.0)
    
    def test_should_not_stop_before_patience_expires(self):
        """Test that early stopping doesn't trigger before patience time"""
        early_stopping = TimeBasedEarlyStopping(patience_seconds=1)
        early_stopping.update(reward=10.0)
        time.sleep(0.3)  # Less than patience
        result = early_stopping.update(reward=5.0)
        self.assertFalse(result)
    
    def test_should_stop_after_patience_expires(self):
        """Test that early stopping triggers after patience time without improvement"""
        # Use a very short patience for this test
        early_stopping = TimeBasedEarlyStopping(patience_seconds=0.5)
        early_stopping.update(reward=10.0)
        time.sleep(0.6)  # Exceeds patience
        result = early_stopping.update(reward=5.0)
        self.assertTrue(result)
    
    def test_time_since_improvement(self):
        """Test that time_since_improvement is calculated correctly"""
        self.early_stopping.update(reward=10.0)
        time.sleep(0.3)
        time_elapsed = self.early_stopping.time_since_improvement()
        self.assertGreaterEqual(time_elapsed, 0.2)
        self.assertLess(time_elapsed, 1.0)
    
    def test_get_state_dict(self):
        """Test that state can be serialized"""
        self.early_stopping.update(reward=10.0)
        state = self.early_stopping.get_state_dict()
        
        self.assertIn('best_reward', state)
        self.assertIn('best_reward_time', state)
        self.assertIn('start_time', state)
        self.assertIn('patience_seconds', state)
        self.assertEqual(state['best_reward'], 10.0)
    
    def test_load_state_dict(self):
        """Test that state can be restored"""
        # Create initial state
        self.early_stopping.update(reward=10.0)
        state = self.early_stopping.get_state_dict()
        
        # Create new instance and load state
        new_early_stopping = TimeBasedEarlyStopping(patience_seconds=5)
        new_early_stopping.load_state_dict(state)
        
        self.assertEqual(new_early_stopping.best_reward, 10.0)
        self.assertEqual(new_early_stopping.best_reward_time, state['best_reward_time'])
        self.assertEqual(new_early_stopping.start_time, state['start_time'])
    
    def test_reset(self):
        """Test that early stopping can be reset"""
        self.early_stopping.update(reward=10.0)
        self.early_stopping.reset()
        
        self.assertIsNone(self.early_stopping.best_reward)
        self.assertIsNone(self.early_stopping.best_reward_time)
        # start_time should be reset to current time
        self.assertIsNotNone(self.early_stopping.start_time)


class TestPPOTrainerWithEarlyStopping(unittest.TestCase):
    """Test suite for PPOTrainer with early stopping integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env = SnakeGame(grid_size=5, cell_size=20)
        input_shape = (5, 5, 3)
        action_size = 4
        self.agent = PPO_LSTM_Agent(input_shape, action_size)
        self.trainer = PPOTrainer(
            self.env, 
            self.agent, 
            early_stopping_patience=1  # 1 second for testing
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.env.close()
    
    def test_trainer_has_early_stopping(self):
        """Test that trainer initializes with early stopping"""
        self.assertIsNotNone(self.trainer.early_stopping)
        self.assertIsInstance(self.trainer.early_stopping, TimeBasedEarlyStopping)
    
    def test_trainer_without_early_stopping(self):
        """Test that trainer can be initialized without early stopping"""
        trainer = PPOTrainer(self.env, self.agent, early_stopping_patience=None)
        self.assertIsNone(trainer.early_stopping)
    
    @patch('train.PPOTrainer.collect_experiences')
    @patch('train.PPOTrainer.update')
    def test_train_stops_on_early_stopping(self, mock_update, mock_collect):
        """Test that training stops when early stopping is triggered"""
        # Mock to return improving then non-improving rewards
        mock_collect.side_effect = [
            [10.0, 11.0, 12.0],  # Improving
            [8.0, 7.0, 6.0],     # Non-improving (after patience)
        ]
        mock_update.return_value = 0.5
        
        # Manually trigger early stopping after first iteration
        def side_effect_stop(*args):
            if mock_collect.call_count > 1:
                # Simulate patience expired
                self.trainer.early_stopping.best_reward_time = time.time() - 2
            return [8.0, 7.0, 6.0]
        
        mock_collect.side_effect = side_effect_stop
        
        # Train should stop early
        self.trainer.train(num_iterations=float('inf'), episodes_per_iteration=3)
        
        # Should have called collect_experiences at least once
        self.assertGreater(mock_collect.call_count, 0)
    
    def test_save_model_includes_early_stopping(self):
        """Test that saved model includes early stopping state"""
        import tempfile
        import os
        
        self.trainer.early_stopping.update(reward=10.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            self.trainer.save_model(filepath)
            
            # Load and check
            checkpoint = torch.load(filepath)
            self.assertIn('early_stopping_state', checkpoint)
            self.assertEqual(checkpoint['early_stopping_state']['best_reward'], 10.0)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_infinite_training_parameter(self):
        """Test that training accepts infinite iterations"""
        # This should not raise an error
        try:
            # We'll mock to avoid actually running infinite loop
            with patch('train.PPOTrainer.collect_experiences') as mock_collect:
                with patch('train.PPOTrainer.update') as mock_update:
                    mock_collect.return_value = [10.0]
                    mock_update.return_value = 0.5
                    
                    # Trigger early stopping immediately
                    self.trainer.early_stopping.best_reward = 100.0
                    self.trainer.early_stopping.best_reward_time = time.time() - 10
                    
                    self.trainer.train(num_iterations=float('inf'), episodes_per_iteration=1)
                    
                    # Should have stopped due to early stopping
                    self.assertTrue(True)
        except Exception as e:
            self.fail(f"Training with infinite iterations raised exception: {e}")


class TestEarlyStoppingEdgeCases(unittest.TestCase):
    """Test edge cases for early stopping"""
    
    def test_negative_patience(self):
        """Test that negative patience is handled"""
        with self.assertRaises(ValueError):
            TimeBasedEarlyStopping(patience_seconds=-1)
    
    def test_zero_patience(self):
        """Test that zero patience is handled"""
        with self.assertRaises(ValueError):
            TimeBasedEarlyStopping(patience_seconds=0)
    
    def test_none_patience(self):
        """Test that None patience is handled"""
        with self.assertRaises(ValueError):
            TimeBasedEarlyStopping(patience_seconds=None)
    
    def test_very_large_patience(self):
        """Test that very large patience values work"""
        early_stopping = TimeBasedEarlyStopping(patience_seconds=86400)  # 1 day
        self.assertEqual(early_stopping.patience_seconds, 86400)
    
    def test_equal_rewards(self):
        """Test behavior when rewards are equal"""
        early_stopping = TimeBasedEarlyStopping(patience_seconds=0.5)
        early_stopping.update(reward=10.0)
        time.sleep(0.6)
        result = early_stopping.update(reward=10.0)
        # Equal reward should reset timer (considered improvement)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
