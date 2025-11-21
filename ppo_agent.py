import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPO_LSTM_Agent(nn.Module):
    def __init__(self, input_shape, action_size, hidden_size=256, lstm_hidden_size=128):
        """
        PPO Agent with LSTM memory for sequential decision making
        
        Args:
            input_shape: shape of the input state (height, width, channels)
            action_size: number of possible actions
            hidden_size: size of hidden layers in CNN
            lstm_hidden_size: size of LSTM hidden state
        """
        super(PPO_LSTM_Agent, self).__init__()
        
        self.input_shape = input_shape
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers before LSTM
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, lstm_hidden_size)
        
        # LSTM layer for memory
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, batch_first=True)
        
        # Actor head (policy)
        self.actor = nn.Linear(lstm_hidden_size, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(lstm_hidden_size, 1)
        
        # Initialize hidden state
        self.hidden_state = None
        
    def _get_conv_output_size(self):
        """Calculate the output size after convolutional layers"""
        # Create a dummy input to pass through conv layers
        dummy_input = torch.zeros(1, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        
        x = F.relu(self.conv1(dummy_input))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        return x.numel()
    
    def forward(self, state, reset_hidden=False):
        """
        Forward pass through the network
        
        Args:
            state: input state tensor of shape (batch_size, height, width, channels)
            reset_hidden: whether to reset LSTM hidden state
            
        Returns:
            action_probs: action probabilities
            value: state value estimate
        """
        batch_size = state.shape[0]
        
        # Reshape state for CNN: (batch_size, channels, height, width)
        state = state.permute(0, 3, 1, 2).float()
        
        # CNN feature extraction
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)  # (batch_size, 1, lstm_hidden_size)
        
        # LSTM layer
        if reset_hidden or self.hidden_state is None:
            # Initialize hidden state
            h0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden_size).to(x.device)
            self.hidden_state = (h0, c0)
        
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Actor and Critic heads
        action_logits = self.actor(x)
        value = self.critic(x)
        
        # Apply softmax to get action probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, value
    
    def reset_hidden_state(self, batch_size=1):
        """Reset the LSTM hidden state"""
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_size)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_size)
        self.hidden_state = (h0, c0)
    
    def select_action(self, state):
        """
        Select action based on current policy
        
        Args:
            state: current state
            
        Returns:
            action: selected action
            log_prob: log probability of the action
            value: state value
        """
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            action_probs, value = self.forward(state, reset_hidden=True)
            
        # Sample action from the probability distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for PPO update
        
        Args:
            states: batch of states
            actions: batch of actions taken
            
        Returns:
            log_probs: log probabilities of actions
            values: state values
            entropy: entropy of the policy
        """
        action_probs, values = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy