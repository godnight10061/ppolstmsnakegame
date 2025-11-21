"""
Script to verify the LSTM hidden state mismatch fix
This reproduces the exact error scenario from the issue
"""
import torch
from snake_game import SnakeGame
from ppo_agent import PPO_LSTM_Agent
from train import PPOTrainer

def test_original_error_scenario():
    """
    Reproduce the original error scenario:
    RuntimeError: Expected hidden[0] size (1, 64, 128), got [1, 1, 128]
    """
    print("Testing original error scenario...")
    print("=" * 60)
    
    # Create environment with standard settings
    env = SnakeGame(grid_size=10, cell_size=30)
    
    # Create agent
    input_shape = (env.grid_size, env.grid_size, 3)
    action_size = 4
    agent = PPO_LSTM_Agent(input_shape, action_size)
    
    # Create trainer with batch_size=64 (as in the error)
    trainer = PPOTrainer(env, agent, batch_size=64, early_stopping_patience=None)
    
    try:
        # This should trigger the exact sequence that caused the error:
        # 1. Collect experiences (calls select_action with batch_size=1)
        # 2. Update (calls evaluate_actions with batch_size=64)
        print("Collecting experiences (uses batch_size=1 for action selection)...")
        episode_rewards = trainer.collect_experiences(num_episodes=2)
        print(f"✓ Collected {len(episode_rewards)} episodes")
        
        print(f"Updating agent (uses batch_size=64 for evaluation)...")
        loss = trainer.update()
        print(f"✓ Update successful, loss: {loss:.4f}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: The LSTM hidden state mismatch error is FIXED!")
        print("=" * 60)
        return True
        
    except RuntimeError as e:
        if "Expected hidden" in str(e):
            print("\n" + "=" * 60)
            print("FAILURE: LSTM hidden state mismatch error still occurs!")
            print(f"Error: {e}")
            print("=" * 60)
            return False
        else:
            raise
    finally:
        env.close()

def test_multiple_batch_sizes():
    """Test with various batch sizes to ensure robustness"""
    print("\n\nTesting with multiple batch sizes...")
    print("=" * 60)
    
    env = SnakeGame(grid_size=10, cell_size=30)
    input_shape = (env.grid_size, env.grid_size, 3)
    action_size = 4
    agent = PPO_LSTM_Agent(input_shape, action_size)
    
    batch_sizes = [16, 32, 64, 128]
    all_passed = True
    
    for batch_size in batch_sizes:
        try:
            trainer = PPOTrainer(env, agent, batch_size=batch_size, 
                               early_stopping_patience=None)
            episode_rewards = trainer.collect_experiences(num_episodes=1)
            loss = trainer.update()
            print(f"✓ batch_size={batch_size:3d}: SUCCESS (loss: {loss:.4f})")
        except RuntimeError as e:
            if "Expected hidden" in str(e):
                print(f"✗ batch_size={batch_size:3d}: FAILED - {e}")
                all_passed = False
            else:
                raise
    
    env.close()
    
    print("=" * 60)
    if all_passed:
        print("SUCCESS: All batch sizes work correctly!")
    else:
        print("FAILURE: Some batch sizes still have issues")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    result1 = test_original_error_scenario()
    result2 = test_multiple_batch_sizes()
    
    if result1 and result2:
        print("\n\n🎉 All verification tests PASSED! 🎉")
        exit(0)
    else:
        print("\n\n❌ Some verification tests FAILED ❌")
        exit(1)
