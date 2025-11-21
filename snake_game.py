import pygame
import numpy as np
from typing import Tuple, List, Optional
import random

class SnakeGame:
    def __init__(self, grid_size=20, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Initialize snake at center
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = 1  # 0: up, 1: right, 2: down, 3: left
        self.food = self._generate_food()
        self.score = 0
        self.steps_without_food = 0
        self.max_steps_without_food = self.grid_size * self.grid_size
        self.done = False
        
        return self.get_state()
    
    def _generate_food(self) -> Tuple[int, int]:
        """Generate food at random position not occupied by snake"""
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                   random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment
        
        Args:
            action: 0: up, 1: right, 2: down, 3: left
            
        Returns:
            state: next state
            reward: reward for this step
            done: whether game is over
            info: additional information
        """
        if self.done:
            return self.get_state(), 0, True, {}
        
        # Update direction (prevent going back into itself)
        if (action == 0 and self.direction != 2) or \
           (action == 1 and self.direction != 3) or \
           (action == 2 and self.direction != 0) or \
           (action == 3 and self.direction != 1):
            self.direction = action
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        if self.direction == 0:  # up
            new_head = (head_x, head_y - 1)
        elif self.direction == 1:  # right
            new_head = (head_x + 1, head_y)
        elif self.direction == 2:  # down
            new_head = (head_x, head_y + 1)
        else:  # left
            new_head = (head_x - 1, head_y)
        
        # Check boundaries
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.done = True
            return self.get_state(), -10, self.done, {"score": self.score}
        
        # Check self collision
        if new_head in self.snake[1:]:
            self.done = True
            return self.get_state(), -10, self.done, {"score": self.score}
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._generate_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()
            self.steps_without_food += 1
            # Small negative reward for each step to encourage efficiency
            reward = -0.01
        
        # End game if too many steps without food
        if self.steps_without_food >= self.max_steps_without_food:
            self.done = True
            reward = -5
        
        return self.get_state(), reward, self.done, {"score": self.score}
    
    def get_state(self) -> np.ndarray:
        """
        Get current state representation
        
        Returns:
            numpy array representing the current state
        """
        # Create grid representation
        state = np.zeros((self.grid_size, self.grid_size, 3))
        
        # Snake body (channel 0)
        for segment in self.snake:
            state[segment[1], segment[0], 0] = 1.0
        
        # Snake head (channel 1)
        head = self.snake[0]
        state[head[1], head[0], 1] = 1.0
        
        # Food (channel 2)
        state[self.food[1], self.food[0], 2] = 1.0
        
        return state
    
    def render(self, mode='human'):
        """Render the game"""
        if mode == 'human':
            self.screen.fill(self.BLACK)
            
            # Draw snake
            for i, segment in enumerate(self.snake):
                color = self.BLUE if i == 0 else self.GREEN
                pygame.draw.rect(self.screen, color,
                               (segment[0] * self.cell_size,
                                segment[1] * self.cell_size,
                                self.cell_size, self.cell_size))
            
            # Draw food
            pygame.draw.rect(self.screen, self.RED,
                           (self.food[0] * self.cell_size,
                            self.food[1] * self.cell_size,
                            self.cell_size, self.cell_size))
            
            # Draw score
            font = pygame.font.Font(None, 36)
            text = font.render(f'Score: {self.score}', True, self.WHITE)
            self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(10)
    
    def close(self):
        """Close the game window"""
        pygame.quit()