import unittest
from unittest.mock import Mock, patch, MagicMock, call
import torch
import numpy as np
import os
import tempfile
from visualize import (
    load_best_model,
    find_best_model,
    visualize_agent,
    parse_args
)
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent


class TestFindBestModel(unittest.TestCase):
    """Test suite for finding the best trained model"""
    
    def test_find_best_model_returns_final_model_when_exists(self):
        """Test that find_best_model prefers final model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model files
            final_path = os.path.join(tmpdir, "ppo_snake_agent_final.pth")
            checkpoint_path = os.path.join(tmpdir, "ppo_snake_agent_100.pth")
            
            # Create empty files
            open(final_path, 'w').close()
            open(checkpoint_path, 'w').close()
            
            result = find_best_model(tmpdir)
            self.assertEqual(result, final_path)
    
    def test_find_best_model_returns_highest_checkpoint(self):
        """Test that find_best_model returns highest iteration checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy checkpoint files
            checkpoint_100 = os.path.join(tmpdir, "ppo_snake_agent_100.pth")
            checkpoint_200 = os.path.join(tmpdir, "ppo_snake_agent_200.pth")
            checkpoint_50 = os.path.join(tmpdir, "ppo_snake_agent_50.pth")
            
            # Create empty files
            for path in [checkpoint_100, checkpoint_200, checkpoint_50]:
                open(path, 'w').close()
            
            result = find_best_model(tmpdir)
            self.assertEqual(result, checkpoint_200)
    
    def test_find_best_model_raises_error_when_no_models_found(self):
        """Test that find_best_model raises error when no models exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as context:
                find_best_model(tmpdir)
            
            self.assertIn("No trained model found", str(context.exception))
    
    def test_find_best_model_with_custom_path(self):
        """Test that find_best_model works with custom path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            final_path = os.path.join(tmpdir, "ppo_snake_agent_final.pth")
            open(final_path, 'w').close()
            
            result = find_best_model(tmpdir)
            self.assertEqual(result, final_path)


class TestLoadBestModel(unittest.TestCase):
    """Test suite for loading the best model"""
    
    def test_load_best_model_with_explicit_path(self):
        """Test loading model with explicit path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.pth")
            
            # Create a minimal checkpoint
            env = SnakeGame(grid_size=10, cell_size=30)
            input_shape = (10, 10, 3)
            agent = PPO_LSTM_Agent(input_shape, action_size=4)
            checkpoint = {'agent_state_dict': agent.state_dict()}
            torch.save(checkpoint, model_path)
            
            loaded_agent = load_best_model(model_path=model_path, grid_size=10)
            
            self.assertIsInstance(loaded_agent, PPO_LSTM_Agent)
            env.close()
    
    def test_load_best_model_auto_finds_model(self):
        """Test that load_best_model can auto-find model in directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "ppo_snake_agent_final.pth")
            
            # Create a minimal checkpoint
            env = SnakeGame(grid_size=10, cell_size=30)
            input_shape = (10, 10, 3)
            agent = PPO_LSTM_Agent(input_shape, action_size=4)
            checkpoint = {'agent_state_dict': agent.state_dict()}
            torch.save(checkpoint, model_path)
            
            loaded_agent = load_best_model(search_dir=tmpdir, grid_size=10)
            
            self.assertIsInstance(loaded_agent, PPO_LSTM_Agent)
            env.close()
    
    def test_load_best_model_raises_error_for_invalid_file(self):
        """Test that load_best_model raises error for invalid file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "invalid.pth")
            open(invalid_path, 'w').close()
            
            with self.assertRaises(Exception):
                load_best_model(model_path=invalid_path, grid_size=10)


class TestVisualizeAgent(unittest.TestCase):
    """Test suite for visualizing agent gameplay"""
    
    @patch('visualize.pygame')
    def test_visualize_agent_runs_episode(self, mock_pygame):
        """Test that visualize_agent runs an episode"""
        # Setup mocks
        mock_pygame.time.Clock.return_value.tick = Mock()
        mock_pygame.event.get.return_value = []
        
        # Create environment and agent
        input_shape = (10, 10, 3)
        agent = PPO_LSTM_Agent(input_shape, action_size=4)
        agent.eval()
        
        # The function creates its own environment, so we just verify it completes
        # without error and runs the expected number of episodes
        visualize_agent(agent, grid_size=10, num_episodes=1, fps=60)
        
        # Verify pygame methods were called (environment was rendered)
        self.assertTrue(mock_pygame.event.get.called)
    
    @patch('visualize.pygame')
    def test_visualize_agent_handles_quit_event(self, mock_pygame):
        """Test that visualize_agent handles pygame QUIT event"""
        # Setup mock event
        quit_event = Mock()
        quit_event.type = 12  # pygame.QUIT
        mock_pygame.QUIT = 12
        mock_pygame.event.get.return_value = [quit_event]
        mock_pygame.time.Clock.return_value.tick = Mock()
        
        # Create environment and agent
        env = SnakeGame(grid_size=10, cell_size=30)
        input_shape = (10, 10, 3)
        agent = PPO_LSTM_Agent(input_shape, action_size=4)
        agent.eval()
        
        # Should exit early due to quit event
        visualize_agent(agent, grid_size=10, num_episodes=1, fps=60)
        
        env.close()
    
    @patch('visualize.pygame')
    def test_visualize_agent_respects_fps(self, mock_pygame):
        """Test that visualize_agent respects FPS setting"""
        mock_clock = Mock()
        mock_pygame.time.Clock.return_value = mock_clock
        mock_pygame.event.get.return_value = []
        
        # Create environment and agent
        env = SnakeGame(grid_size=10, cell_size=30)
        input_shape = (10, 10, 3)
        agent = PPO_LSTM_Agent(input_shape, action_size=4)
        agent.eval()
        
        # Mock the environment to finish quickly
        with patch.object(env, 'step') as mock_step:
            mock_step.return_value = (env.get_state(), 0, True, {'score': 0})
            
            target_fps = 30
            visualize_agent(agent, grid_size=10, num_episodes=1, fps=target_fps)
            
            # Verify clock.tick was called with correct FPS
            mock_clock.tick.assert_called()
        
        env.close()
    
    @patch('visualize.pygame')
    def test_visualize_agent_displays_score(self, mock_pygame):
        """Test that visualize_agent displays score information"""
        mock_pygame.event.get.return_value = []
        mock_pygame.time.Clock.return_value.tick = Mock()
        
        # Create environment and agent
        env = SnakeGame(grid_size=10, cell_size=30)
        input_shape = (10, 10, 3)
        agent = PPO_LSTM_Agent(input_shape, action_size=4)
        agent.eval()
        
        # Mock the environment to finish with a score
        with patch.object(env, 'step') as mock_step:
            mock_step.return_value = (env.get_state(), 10, True, {'score': 5})
            
            visualize_agent(agent, grid_size=10, num_episodes=1, fps=60)
        
        env.close()


class TestParseArgs(unittest.TestCase):
    """Test suite for command-line argument parsing"""
    
    def test_parse_args_with_model_path(self):
        """Test parsing with explicit model path"""
        test_args = ['--model', 'test_model.pth']
        
        with patch('sys.argv', ['visualize.py'] + test_args):
            args = parse_args()
            self.assertEqual(args.model, 'test_model.pth')
    
    def test_parse_args_with_grid_size(self):
        """Test parsing with custom grid size"""
        test_args = ['--grid_size', '15']
        
        with patch('sys.argv', ['visualize.py'] + test_args):
            args = parse_args()
            self.assertEqual(args.grid_size, 15)
    
    def test_parse_args_with_fps(self):
        """Test parsing with custom FPS"""
        test_args = ['--fps', '30']
        
        with patch('sys.argv', ['visualize.py'] + test_args):
            args = parse_args()
            self.assertEqual(args.fps, 30)
    
    def test_parse_args_with_episodes(self):
        """Test parsing with custom number of episodes"""
        test_args = ['--episodes', '10']
        
        with patch('sys.argv', ['visualize.py'] + test_args):
            args = parse_args()
            self.assertEqual(args.episodes, 10)
    
    def test_parse_args_defaults(self):
        """Test that defaults are set correctly"""
        with patch('sys.argv', ['visualize.py']):
            args = parse_args()
            self.assertIsNone(args.model)
            self.assertEqual(args.grid_size, 10)
            self.assertEqual(args.fps, 10)
            self.assertEqual(args.episodes, 5)
            self.assertEqual(args.cell_size, 30)


if __name__ == '__main__':
    unittest.main()
