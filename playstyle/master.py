import numpy as np
from .base_playstyle import BasePlayStyle
from game_logic import Game2048
import random

class MasterPlayStyle(BasePlayStyle):
    def __init__(self, game_config, gen_config):
        super().__init__(game_config, gen_config)
        self.preferred_corner = (3, 0) # Bottom-left corner
        self.snake_pattern = [
            (3, 0), (3, 1), (3, 2), (3, 3),
            (2, 3), (2, 2), (2, 1), (2, 0),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (0, 3), (0, 2), (0, 1), (0, 0)
        ]
        self.move_priorities = {
            'build_snake': ['left', 'down', 'right', 'up'],
            'merge_tiles': ['down', 'left', 'right', 'up'],
            'emergency': ['up', 'right', 'down', 'left']
        }

    def get_monotonicity_score(self, board):
        """Calculate how well the board follows a monotonic pattern"""
        score = 0
        # Check rows with exponential weighting
        for i in range(4):
            for j in range(3):
                if board[i][j] >= board[i][j + 1] and board[i][j] != 0:
                    score += 2 ** (np.log2(board[i][j]))
        # Check columns with exponential weighting
        for j in range(4):
            for i in range(3):
                if board[i][j] <= board[i + 1][j] and board[i + 1][j] != 0:
                    score += 2 ** (np.log2(board[i + 1][j]))
        return score

    def get_snake_score(self, board):
        """Calculate how well the board follows the snake pattern"""
        score = 0
        max_tile = np.max(board)
        values = []
        for i, j in self.snake_pattern:
            values.append(board[i][j])
        
        for i in range(len(values) - 1):
            if values[i] >= values[i + 1] and values[i] != 0:
                score += 2 ** (np.log2(values[i]))
            elif values[i] != 0:
                penalty = np.log2(max_tile) - np.log2(values[i])
                score -= penalty * 100
        
        # Bonus for max tile in preferred corner
        if board[self.preferred_corner] == max_tile:
            score += max_tile * 4

        return score

    def get_merge_opportunities(self, board):
        """Count potential merges weighted by tile values"""
        score = 0
        # Horizontal merges with exponential weighting
        for i in range(4):
            for j in range(3):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    score += 2 ** (np.log2(board[i][j]) + 1)
        # Vertical merges with exponential weighting
        for i in range(3):
            for j in range(4):
                if board[i][j] == board[i + 1][j] and board[i][j] != 0:
                    score += 2 ** (np.log2(board[i][j]) + 1)
        return score

    def get_empty_cells_score(self, board):
        """Calculate score based on empty cells and their positions"""
        empty_count = len(np.where(board == 0)[0])
        empty_score = empty_count * 500

        max_tile = np.max(board)
        # Bonus for empty cells near high values
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 4 and 0 <= nj < 4 and board[ni][nj] != 0:
                            empty_score += 2 ** (np.log2(board[ni][nj]))

        return empty_score

    def evaluate_position(self, board, phase):
        """Evaluate board position with different weights based on game phase"""
        # Base scores with adjusted weights
        monotonicity = self.get_monotonicity_score(board)
        snake_score = self.get_snake_score(board)
        merge_score = self.get_merge_opportunities(board)
        empty_score = self.get_empty_cells_score(board)
        
        # Dynamic weights based on game phase
        if phase == 'early':
            return (empty_score * 3.0 + 
                   merge_score * 2.0 + 
                   monotonicity * 1.0 + 
                   snake_score * 0.5)
        elif phase == 'mid':
            return (empty_score * 2.0 + 
                   merge_score * 1.5 + 
                   monotonicity * 2.0 + 
                   snake_score * 2.0)
        else:  # late game
            return (empty_score * 1.5 + 
                   merge_score * 1.0 + 
                   monotonicity * 2.5 + 
                   snake_score * 3.0)

    def determine_game_phase(self, board):
        """Determine game phase based on max tile and board fullness"""
        max_tile = np.max(board)
        empty_cells = len(np.where(board == 0)[0])
        
        if max_tile < 256 or empty_cells >= 8:
            return 'early'
        elif max_tile < 1024 or empty_cells >= 4:
            return 'mid'
        else:
            return 'late'

    def get_move_sequence(self, phase, board):
        """Get optimal move sequence based on game phase and board state"""
        if phase == 'early':
            return self.move_priorities['merge_tiles']
        elif phase == 'mid':
            if self.get_snake_score(board) > 0:
                return self.move_priorities['build_snake']
            return self.move_priorities['merge_tiles']
        else:
            return self.move_priorities['build_snake']

    def simulate_moves(self, game, moves_sequence, depth=3):
        """Simulate a sequence of moves and evaluate resulting position"""
        best_score = float('-inf')
        best_move = None
        
        for first_move in moves_sequence:
            current_score = 0
            test_game = Game2048(config_dict=self.game_config)
            test_game.board = game.board.copy()
            
            # Try first move
            original = test_game.board.copy()
            test_game.move(first_move)
            if np.array_equal(original, test_game.board):
                continue
            
            # Simulate random next moves
            for _ in range(depth):
                sim_game = Game2048(config_dict=self.game_config)
                sim_game.board = test_game.board.copy()
                
                for _ in range(3):
                    move = random.choice(moves_sequence)
                    sim_game.move(move)
                
                phase = self.determine_game_phase(sim_game.board)
                current_score += self.evaluate_position(sim_game.board, phase)
            
            if current_score > best_score:
                best_score = current_score
                best_move = first_move
        
        return best_move

    def generate_move(self, game):
        """Generate the next move using advanced strategies"""
        board = game.board
        phase = self.determine_game_phase(board)
        moves_sequence = self.get_move_sequence(phase, board)
        
        # Try planned sequence first
        move = self.simulate_moves(game, moves_sequence)
        
        # If no good move found, try emergency sequence
        if move is None:
            move = self.simulate_moves(game, self.move_priorities['emergency'])
        
        # If still no move found, try any possible move
        if move is None:
            for move in ['up', 'down', 'left', 'right']:
                test_game = Game2048(config_dict=self.game_config)
                test_game.board = game.board.copy()
                original = test_game.board.copy()
                test_game.move(move)
                if not np.array_equal(original, test_game.board):
                    return move
        
        return move
