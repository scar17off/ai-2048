import numpy as np
import random
from .base_playstyle import BasePlayStyle
from game_logic import Game2048

class MonteCarloPlayStyle(BasePlayStyle):
    def evaluate_position(self, board):
        score = 0
        max_tile = np.max(board)
        grid_size = board.shape
        
        # Create weight matrix based on grid size
        weight_matrix = np.zeros_like(board, dtype=float)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                weight_matrix[i][j] = grid_size[0] * grid_size[1] - (i * grid_size[1] + j)
        
        # Calculate weighted sum
        score += np.sum(board * weight_matrix) * 2
        
        # Count empty cells
        empty_count = len(np.where(board == 0)[0])
        score += empty_count * 100
        
        # Bonus for max tile in corner
        corners = [(0,0), (0,grid_size[1]-1), (grid_size[0]-1,0), (grid_size[0]-1,grid_size[1]-1)]
        for i, j in corners:
            if board[i][j] == max_tile:
                score += max_tile * 4
                break
        
        return score

    def run_random_simulation(self, game, first_move, num_simulations=None):
        if num_simulations is None:
            mc_config = self.gen_config['training']['monte_carlo']
            max_tile = np.max(game.board)
            if max_tile < mc_config['early_game_threshold']:
                num_simulations = mc_config['simulations_early_game']
            elif max_tile < mc_config['mid_game_threshold']:
                num_simulations = mc_config['simulations_mid_game']
            else:
                num_simulations = mc_config['simulations_late_game']
        
        test_game = Game2048(config_dict=self.game_config)
        test_game.board = game.board.copy()
        
        # Make the first move
        original_board = test_game.board.copy()
        test_game.move(first_move)
        
        if np.array_equal(original_board, test_game.board):
            return float('-inf')
        
        total_score = 0
        max_reached = 0
        
        for _ in range(num_simulations):
            sim_game = Game2048(config_dict=self.game_config)
            sim_game.board = test_game.board.copy()
            moves_without_progress = 0
            last_max = np.max(sim_game.board)
            
            while not sim_game.is_game_over():
                move = random.choice(['up', 'down', 'left', 'right'])
                original = sim_game.board.copy()
                sim_game.move(move)
                
                current_max = np.max(sim_game.board)
                if current_max > last_max:
                    moves_without_progress = 0
                    last_max = current_max
                else:
                    moves_without_progress += 1
                    
                if moves_without_progress > 10 or np.array_equal(original, sim_game.board):
                    break
            
            total_score += sim_game.score
            max_reached = max(max_reached, np.max(sim_game.board))
        
        return total_score / num_simulations + max_reached

    def generate_move(self, game):
        best_score = float('-inf')
        best_move = None
        max_tile = np.max(game.board)
        mc_config = self.gen_config['training']['monte_carlo']
        
        if max_tile < mc_config['early_game_threshold']:
            num_sims = mc_config['simulations_early_game']
        elif max_tile < mc_config['mid_game_threshold']:
            num_sims = mc_config['simulations_mid_game']
        else:
            num_sims = mc_config['simulations_late_game']
        
        for move in ['up', 'left', 'down', 'right']:
            test_game = Game2048(config_dict=self.game_config)
            test_game.board = game.board.copy()
            original = test_game.board.copy()
            test_game.move(move)
            
            if np.array_equal(original, test_game.board):
                continue
            
            position_score = self.evaluate_position(test_game.board)
            simulation_score = self.run_random_simulation(game, move, num_sims)
            total_score = position_score + simulation_score
            
            if total_score > best_score:
                best_score = total_score
                best_move = move
        
        return best_move
