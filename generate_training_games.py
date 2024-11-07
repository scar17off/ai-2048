import numpy as np
import os
from game_logic import Game2048
from datetime import datetime
import time
import sys
import multiprocessing as mp
from multiprocessing import Pool, Manager
import json
from playstyle.monte_carlo import MonteCarloPlayStyle
from playstyle.master import MasterPlayStyle

with open('generation_config.json', 'r') as f:
    GEN_CONFIG = json.load(f)

with open('config.json', 'r') as f:
    GAME_CONFIG = json.load(f)

def dual_progress_bar(game_progress, total_progress, game_scores, attempts, success_rate=None, eta=None):
    width = 50
    
    # Clear previous lines for all workers
    sys.stdout.write('\033[K' * (len(game_scores) + 1))
    sys.stdout.write('\033[F' * (len(game_scores)))
    
    # Print progress bar for each worker
    for worker_id, score in game_scores.items():
        game_percentage = min(100, (np.log2(max(score, 2)) / 11.0) * 100) # log2(2048) = 11
        game_filled = int(width * game_percentage / 100)
        game_bar = '=' * (game_filled - 1) + '>' if game_filled > 0 else ''
        game_bar = game_bar.ljust(width, '.')
        sys.stdout.write(f"\rWorker {worker_id} [{game_bar}] {score}/2048\n")
    
    # Total progress bar
    total_filled = int(width * total_progress)
    total_bar = '=' * (total_filled - 1) + '>' if total_filled > 0 else ''
    total_bar = total_bar.ljust(width, '.')
    
    # Status information
    eta_str = f"ETA: {eta:.1f}s" if eta is not None else "ETA: calculating..."
    attempts_str = f"Attempts: {attempts.value}" if attempts is not None else ""
    
    sys.stdout.write(f"\rTotal [{total_bar}] {eta_str} {attempts_str}")
    sys.stdout.flush()

def get_empty_cells(board):
    return len(np.where(board == 0)[0])

def get_mergeable_tiles(board):
    count = 0
    # Check horizontal merges
    for i in range(4):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] == board[i][j + 1]:
                count += 1
    # Check vertical merges
    for i in range(3):
        for j in range(4):
            if board[i][j] != 0 and board[i][j] == board[i + 1][j]:
                count += 1
    return count

def board_to_string(board):
    return '\n'.join([' '.join(map(str, row)) for row in board])

def save_game_to_txt(game_history, score, filename, max_tile):
    with open(filename, 'w') as f:
        f.write(f"Final Score: {score}\n")
        f.write(f"Max Tile: {max_tile}\n")
        f.write(f"Number of Moves: {len(game_history)}\n\n")
        
        for i, (board, move) in enumerate(game_history, 1):
            f.write(f"Move {i}:\n")
            f.write(board_to_string(board))
            f.write(f"\nAction: {move}\n\n")

def get_playstyle(game_config, gen_config):
    playstyle_name = gen_config['training']['playstyle']
    playstyles = {
        'monte_carlo': MonteCarloPlayStyle,
        'master': MasterPlayStyle
    }
    
    if playstyle_name not in playstyles:
        raise ValueError(f"Unknown playstyle: {playstyle_name}")
    
    return playstyles[playstyle_name](game_config, gen_config)

def worker_generate_game(worker_id, shared_dict, shared_attempts, shared_successful, lock):
    game = Game2048(config_dict=GAME_CONFIG)
    playstyle = get_playstyle(GAME_CONFIG, GEN_CONFIG)
    game_history = []
    moves_without_progress = GEN_CONFIG['training']['early_termination']['moves_without_progress']
    consecutive_same_moves = 0
    last_max = 0
    last_move = None
    temp_filename = None
    highest_saved_score = 0
    
    while not game.is_game_over():
        current_state = game.get_state()
        current_max = np.max(current_state)
        
        # Update shared dictionary with current score
        with lock:
            shared_dict[worker_id] = current_max
        
        # Progress tracking
        if current_max > last_max:
            moves_without_progress = 0
            consecutive_same_moves = 0
            
            # Save game if we've reached a new milestone and it's above minimum threshold
            if current_max >= GEN_CONFIG['training']['min_score_to_save'] and current_max > highest_saved_score:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_filename = f'games/game_{current_max}_{game.score}_{timestamp}.txt'
                save_game_to_txt(game_history, game.score, new_filename, current_max)
                
                # Remove previous temporary save if it exists
                if temp_filename and os.path.exists(temp_filename):
                    os.remove(temp_filename)
                
                temp_filename = new_filename
                highest_saved_score = current_max
            
            last_max = current_max
        else:
            moves_without_progress += 1
        
        # Early termination conditions from config
        if moves_without_progress > GEN_CONFIG['training']['early_termination']['moves_without_progress']:
            return [], 0, False
        
        move = playstyle.generate_move(game)
        if move is None:
            break
        
        # Detect repetitive moves
        if move == last_move:
            consecutive_same_moves += 1
            if consecutive_same_moves > GEN_CONFIG['training']['early_termination']['consecutive_same_moves']:
                return [], 0, False
        else:
            consecutive_same_moves = 0
        
        last_move = move
        game_history.append((current_state.copy(), move))
        game.move(move)
        
        if current_max >= GAME_CONFIG['game']['win_tile']:
            return game_history, game.score, True
    
    return game_history, game.score, False

def generate_games_parallel(num_workers=None):
    worker_config = GEN_CONFIG['training']['workers']
    
    if num_workers is None:
        if worker_config['mode'] == 'auto':
            num_workers = min(mp.cpu_count(), worker_config['max_workers'])
        else:
            num_workers = worker_config['count']
    
    os.makedirs('games', exist_ok=True)
    
    # Create manager for shared variables
    manager = Manager()
    shared_dict = manager.dict() # For tracking current scores
    shared_attempts = manager.Value('i', 0)
    shared_successful = manager.Value('i', 0)
    lock = manager.Lock() # Create a single lock for synchronization
    
    # Initialize shared dictionary with zeros
    for i in range(num_workers):
        shared_dict[i] = 0
    
    print(f"Generating games that reach 2048 using {num_workers} workers...")
    print("\n" * (num_workers + 1)) # Space for progress bars
    
    start_time = time.time()
    target_successes = GEN_CONFIG['training']['target_games']
    max_attempts = GEN_CONFIG['training']['max_attempts']
    min_score_to_save = GEN_CONFIG['training']['min_score_to_save']
    
    with Pool(num_workers) as pool:
        while shared_successful.value < target_successes and shared_attempts.value < max_attempts:
            # Create worker tasks (args for worker_generate_game)
            tasks = [(i, shared_dict, shared_attempts, shared_successful, lock) 
                    for i in range(num_workers)]
            
            # Run workers asynchronously
            results = [pool.apply_async(worker_generate_game, t) for t in tasks]
            
            # Process results as they complete
            for i, result in enumerate(results):
                game_history, score, reached_2048 = result.get()
                
                # Use the lock for updating shared values
                with lock:
                    shared_attempts.value += 1
                    if reached_2048:
                        shared_successful.value += 1
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'games/game_2048_{score}_{timestamp}.txt'
                        save_game_to_txt(game_history, score, filename, np.max(game_history[-1][0]))
                
                # Update progress display
                current_time = time.time()
                if shared_successful.value > 0:
                    time_per_success = (current_time - start_time) / shared_successful.value
                    remaining = target_successes - shared_successful.value
                    eta = time_per_success * remaining
                else:
                    eta = None
                
                total_progress = shared_successful.value / target_successes
                dual_progress_bar(0, total_progress, shared_dict, 
                                shared_attempts, eta)
    
    print("\n" * 2) # Clear space after progress bars
    
    # Print final statistics
    total_time = time.time() - start_time
    print(f"Successfully generated {shared_successful.value} games that reached 2048!")
    print(f"Total time: {total_time:.1f} seconds")
    if shared_successful.value > 0:
        print(f"Average time per successful game: {total_time/shared_successful.value:.1f} seconds")
    print(f"Final success rate: {(shared_successful.value/shared_attempts.value)*100:.1f}%")

if __name__ == '__main__':
    generate_games_parallel()
