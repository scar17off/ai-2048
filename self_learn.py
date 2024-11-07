import os
import tensorflow as tf
import numpy as np
from game_logic import Game2048
import json
from datetime import datetime
import sys
from pathlib import Path
import logging
import pygame
from colors import COLORS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure GPU settings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logging.info(f"GPU configured successfully: {gpus[0]}")
except Exception as e:
    logging.error(f"Error configuring GPU: {str(e)}")

class SelfLearner:
    def __init__(self, visual_mode=False):
        logging.info("Initializing SelfLearner...")
        self.visual_mode = visual_mode
        
        # Load configurations first
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
            logging.info("Configuration loaded successfully")
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            raise
        
        # Initialize pygame if visual mode is enabled
        if self.visual_mode:
            pygame.init()
            # Get grid size from config
            self.grid_size = self.config['game']['grid_size']
            self.width = 850
            self.height = 400 + (100 if self.grid_size[0] > 4 else 0)
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('2048 AI Training')
            self.cell_size = min(400 // max(self.grid_size), 50)
            self.board_offset = ((400 - (self.cell_size * self.grid_size[0])) // 2,
                               (400 - (self.cell_size * self.grid_size[1])) // 2)
            self.font = pygame.font.Font(None, min(36, self.cell_size))
            self.clock = pygame.time.Clock()
            
            # Initialize matplotlib figure for network visualization
            self.fig, self.ax = plt.subplots(figsize=(4, 4))
            self.fig.patch.set_alpha(0.5)
        
        Path("games").mkdir(exist_ok=True)
        logging.info("Games directory checked/created")
        
        # Initialize model
        logging.info("Loading AI model...")
        try:
            self.model = tf.keras.models.load_model('models/2048_model_final.h5',
                                                  custom_objects={'custom_loss': 'categorical_crossentropy'})
            logging.info("AI model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
        self.game = Game2048(config_dict=self.config)
        self.moves_history = []
        self.scores_history = []
        logging.info("SelfLearner initialization complete")
        
    def update_network_visualization(self, predictions):
        """Update the network visualization with new predictions"""
        if not self.visual_mode:
            return
            
        self.ax.clear()
        moves = ['Up', 'Down', 'Left', 'Right']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        bars = self.ax.bar(moves, predictions, color=colors)
        
        self.ax.set_ylim(0, 1)
        self.ax.set_title('AI Move Confidence')
        
        # Percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height*100:.1f}%',
                        ha='center', va='bottom')
        
        self.fig.canvas.draw()

    def draw_network_visualization(self):
        """Draw the network visualization on pygame surface"""
        if not self.visual_mode:
            return
            
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        
        # Draw on screen
        self.screen.blit(surf, (450, 50))

    def get_ai_move(self):
        """Get the next move from the AI model"""
        state = self.game.board.copy()
        state = np.where(state > 0, np.log2(state), 0).astype(np.float32)
        state = state / 11.0
        
        # Ensure state is 4x4 for the model
        if state.shape != (4, 4):
            rows = np.array_split(state, 4, axis=0)
            reduced_state = np.zeros((4, 4))
            for i, row_group in enumerate(rows):
                cols = np.array_split(row_group, 4, axis=1)
                for j, block in enumerate(cols):
                    reduced_state[i, j] = np.max(block)
            state = reduced_state
        
        state = state.reshape(1, 4, 4, 1)
        predictions = self.model.predict(state, verbose=0)[0]
        
        if self.visual_mode:
            self.update_network_visualization(predictions)
        
        # Try moves in order of confidence
        moves = ['up', 'down', 'left', 'right']
        move_probs = list(zip(moves, predictions))
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        for move, prob in move_probs:
            test_game = Game2048(config_dict=self.config)
            test_game.board = self.game.board.copy()
            original = test_game.board.copy()
            test_game.move(move)
            if not np.array_equal(original, test_game.board):
                logging.debug(f"Selected move: {move} (confidence: {prob:.3f})")
                return move
        
        logging.debug("No valid moves available")
        return None

    def save_game(self, game_number):
        """Save the game moves and score to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{np.max(self.game.board)}_{self.game.score}_{timestamp}.txt"
        filepath = Path("games") / filename
        
        try:
            with open(filepath, 'w') as f:
                # Write header information
                f.write(f"Final Score: {self.game.score}\n")
                f.write(f"Max Tile: {np.max(self.game.board)}\n")
                f.write(f"Number of Moves: {len(self.moves_history)}\n\n")
                
                # Write each move with board state
                for i, move in enumerate(self.moves_history, 1):
                    f.write(f"Move {i}:\n")
                    
                    board_state = self.game.board_states[i-1] # We need to track board states in game
                    
                    for row in board_state:
                        f.write(" ".join(str(int(cell)) for cell in row) + "\n")
                    
                    f.write(f"Action: {move}\n\n")
                
            logging.info(f"Game saved successfully to {filepath}")
            logging.info(f"Game stats - Score: {self.game.score}, Max Tile: {np.max(self.game.board)}, Moves: {len(self.moves_history)}")
        except Exception as e:
            logging.error(f"Error saving game: {str(e)}")
            raise
        
        return filepath

    def print_progress(self, current, total, max_tile):
        """Display progress bar with current status"""
        bar_length = 40
        filled_length = int(bar_length * current // total)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        progress_msg = f'Generating Games: [{bar}] {current}/{total} games | Max Tile: {max_tile}'
        sys.stdout.write(f'\r{progress_msg}')
        sys.stdout.flush()
        
        # Log progress at certain intervals
        if current % 10 == 0 or current == total:
            logging.info(f"Progress: {current}/{total} games completed. Current max tile: {max_tile}")

    def draw_game(self):
        """Draw the current game state using pygame"""
        if not self.visual_mode:
            return
            
        self.screen.fill((250, 248, 239))
        
        # Draw game board
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                value = self.game.board[i][j]
                color = COLORS.get(value, (205, 193, 180))
                pygame.draw.rect(self.screen, color,
                               (self.board_offset[0] + j * self.cell_size, 
                                self.board_offset[1] + i * self.cell_size,
                                self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (187, 173, 160),
                               (self.board_offset[0] + j * self.cell_size, 
                                self.board_offset[1] + i * self.cell_size,
                                self.cell_size, self.cell_size), 2)
                
                if value != 0:
                    text = self.font.render(str(value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=(
                        self.board_offset[0] + j * self.cell_size + self.cell_size // 2,
                        self.board_offset[1] + i * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
        
        # Draw stats below the board
        stats_y = self.board_offset[1] + self.grid_size[0] * self.cell_size + 20
        score_text = self.font.render(f'Score: {self.game.score}', True, (0, 0, 0))
        moves_text = self.font.render(f'Moves: {len(self.moves_history)}', True, (0, 0, 0))
        max_text = self.font.render(f'Max Tile: {np.max(self.game.board)}', True, (0, 0, 0))
        
        self.screen.blit(score_text, (10, stats_y))
        self.screen.blit(moves_text, (10, stats_y + 30))
        self.screen.blit(max_text, (200, stats_y))
        
        # Draw neural network visualization
        self.draw_network_visualization()
        
        pygame.display.flip()
        self.clock.tick(20) # Limit to 20 FPS
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def generate_dataset(self, num_games=100, target_tile=2048):
        """Generate AI gameplay dataset"""
        successful_games = 0
        logging.info(f"Starting dataset generation: {num_games} games (target: {target_tile})")
        print(f"\nGenerating {num_games} games (target tile: {target_tile})...")
        
        try:
            for game_num in range(num_games):
                self.game = Game2048(config_dict=self.config)
                self.moves_history = []
                self.game.board_states = [self.game.board.copy()]
                game_over = False
                
                logging.info(f"Starting game {game_num + 1}")
                while not game_over:
                    if self.visual_mode:
                        self.draw_game()
                    
                    move = self.get_ai_move()
                    if move is None:
                        game_over = True
                        logging.info(f"Game {game_num + 1} over - No valid moves")
                        continue
                    
                    self.moves_history.append(move)
                    self.game.move(move)
                    self.game.board_states.append(self.game.board.copy())
                    
                    current_max = np.max(self.game.board)
                    if current_max >= target_tile:
                        filepath = self.save_game(successful_games + 1)
                        successful_games += 1
                        logging.info(f"Game {game_num + 1} successful! Reached {target_tile}")
                        print(f"\nGame {successful_games}: Reached {target_tile}! Saved to {filepath}")
                        break
                
                self.print_progress(game_num + 1, num_games, np.max(self.game.board))
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            if self.visual_mode:
                pygame.quit()
        
        summary_msg = (f"\n\nDataset Generation Summary:\n"
                      f"Total games attempted: {num_games}\n"
                      f"Successful games (reached {target_tile}): {successful_games}")
        print(summary_msg)
        logging.info(summary_msg)
        return successful_games

if __name__ == '__main__':
    print("=" * 50)
    print("2048 AI Self-Learning Dataset Generator")
    print("=" * 50)
    
    logging.info("Starting Self-Learning Dataset Generator")
    
    try:
        visual = input("Enable visual mode? (y/n, default: n): ").lower() == 'y'
        learner = SelfLearner(visual_mode=visual)
        
        num_games = input("Number of games to generate (default 100): ")
        num_games = int(num_games) if num_games.isdigit() else 100
        
        target = input("Target tile to reach (default 2048): ")
        target = int(target) if target.isdigit() else 2048
        
        logging.info(f"Parameters set - Games: {num_games}, Target: {target}, Visual: {visual}")
        print("\nStarting dataset generation...")
        learner.generate_dataset(num_games, target)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
