import unittest
import torch
import numpy as np
from ppo_agent import PPO_LSTM_Agent


class TestPPOLSTMHiddenState(unittest.TestCase):
    """Tests for LSTM hidden state management in PPO agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_shape = (10, 10, 3)
        self.action_size = 4
        self.agent = PPO_LSTM_Agent(self.input_shape, self.action_size)
        
    def test_evaluate_actions_after_select_action(self):
        """Test that evaluate_actions works correctly after select_action calls"""
        # Simulate action selection (batch_size=1)
        state = np.random.rand(10, 10, 3)
        action, log_prob, value = self.agent.select_action(state)
        
        # Now evaluate a batch of actions (batch_size=64)
        batch_size = 64
        states = torch.FloatTensor(np.random.rand(batch_size, 10, 10, 3))
        actions = torch.LongTensor(np.random.randint(0, 4, size=(batch_size,)))
        
        # This should not raise an error
        log_probs, values, entropy = self.agent.evaluate_actions(states, actions)
        
        # Check output shapes
        self.assertEqual(log_probs.shape, (batch_size,))
        self.assertEqual(values.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))
        
    def test_evaluate_actions_different_batch_sizes(self):
        """Test that evaluate_actions works with different batch sizes"""
        batch_sizes = [1, 16, 32, 64, 128]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                states = torch.FloatTensor(np.random.rand(batch_size, 10, 10, 3))
                actions = torch.LongTensor(np.random.randint(0, 4, size=(batch_size,)))
                
                # This should work for any batch size
                log_probs, values, entropy = self.agent.evaluate_actions(states, actions)
                
                # Check output shapes
                self.assertEqual(log_probs.shape, (batch_size,))
                self.assertEqual(values.shape, (batch_size,))
                self.assertEqual(entropy.shape, (batch_size,))
    
    def test_hidden_state_independence_in_evaluate(self):
        """Test that evaluate_actions doesn't depend on previous hidden state"""
        batch_size = 64
        states = torch.FloatTensor(np.random.rand(batch_size, 10, 10, 3))
        actions = torch.LongTensor(np.random.randint(0, 4, size=(batch_size,)))
        
        # Call evaluate_actions twice - should give same results (deterministic)
        with torch.no_grad():
            # First call
            log_probs1, values1, entropy1 = self.agent.evaluate_actions(states, actions)
            
            # Select some actions in between (changes hidden state to batch_size=1)
            state = np.random.rand(10, 10, 3)
            self.agent.select_action(state)
            
            # Second call - should still work
            log_probs2, values2, entropy2 = self.agent.evaluate_actions(states, actions)
        
        # Results should be the same (evaluate doesn't depend on previous state)
        torch.testing.assert_close(log_probs1, log_probs2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(values1, values2, rtol=1e-5, atol=1e-5)
        
    def test_sequential_select_actions(self):
        """Test that sequential select_action calls work correctly"""
        state = np.random.rand(10, 10, 3)
        
        # Multiple sequential calls should work
        for _ in range(10):
            action, log_prob, value = self.agent.select_action(state)
            self.assertIsInstance(action, int)
            self.assertIsInstance(log_prob, float)
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_size)
    
    def test_mixed_select_and_evaluate(self):
        """Test mixed calls to select_action and evaluate_actions"""
        state = np.random.rand(10, 10, 3)
        batch_size = 32
        
        # Pattern: select -> evaluate -> select -> evaluate
        for _ in range(3):
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Evaluate batch
            states = torch.FloatTensor(np.random.rand(batch_size, 10, 10, 3))
            actions = torch.LongTensor(np.random.randint(0, 4, size=(batch_size,)))
            log_probs, values, entropy = self.agent.evaluate_actions(states, actions)
            
            # Check shapes
            self.assertEqual(log_probs.shape, (batch_size,))
            self.assertEqual(values.shape, (batch_size,))


if __name__ == '__main__':
    unittest.main()
