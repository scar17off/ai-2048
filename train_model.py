import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import json

# Load both configuration files
with open('generation_config.json', 'r') as f:
    GEN_CONFIG = json.load(f)

with open('config.json', 'r') as f:
    CONFIG = json.load(f)

GRID_SIZE = CONFIG['game']['grid_size']

def load_game_data(file_path):
    """Load and parse a game file"""
    states = []
    actions = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    current_board = []
    reading_board = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Action:"):
            action = line.split(":")[1].strip()
            actions.append(action)
            # Verify board dimensions match config
            if len(current_board) == GRID_SIZE[0] and all(len(row) == GRID_SIZE[1] for row in current_board):
                states.append(np.array(current_board))
            else:
                print(f"Warning: Skipping move in {file_path} - incorrect board dimensions")
            current_board = []
            reading_board = False
        elif line[0].isdigit() or line[0] == '0':
            # Reading board line
            board_line = [int(x) for x in line.split()]
            current_board.append(board_line)
            reading_board = True
    
    return states, actions

def preprocess_data(states, actions):
    """Convert states and actions to neural network format"""
    processed_states = []
    
    for state in states:
        # Convert to numpy array if not already
        state = np.array(state)
        
        # Verify state dimensions
        if state.shape != tuple(GRID_SIZE):
            print(f"Warning: Skipping state with incorrect dimensions: {state.shape}")
            continue
        
        # Resize state to 4x4 if necessary using max pooling
        if GRID_SIZE != [4, 4]:
            rows = np.array_split(state, 4, axis=0)
            reduced_state = np.zeros((4, 4))
            for i, row_group in enumerate(rows):
                cols = np.array_split(row_group, 4, axis=1)
                for j, block in enumerate(cols):
                    reduced_state[i, j] = np.max(block)
            state = reduced_state
        
        # Convert to log2 scale (with special handling for 0)
        state = np.where(state > 0, np.log2(state), 0).astype(np.float32)
        state = state / 11.0  # Normalize by log2(2048)
        processed_states.append(state)
    
    X = np.array(processed_states)
    
    # Convert actions to one-hot encoding
    action_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    y = np.array([action_map[a] for a in actions])
    y = tf.keras.utils.to_categorical(y, 4)
    
    return X, y

def create_model():
    """Create the neural network model"""
    model = models.Sequential([
        layers.Input(shape=(4, 4, 1)),
        
        layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        layers.Conv2D(64, (2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
        layers.Conv2D(128, (2, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=GEN_CONFIG['model']['learning_rate']
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    os.makedirs('models', exist_ok=True)
    
    print(f"Using grid size from config: {GRID_SIZE[0]}x{GRID_SIZE[1]}")
    
    game_files = glob.glob('games/*.txt')
    print(f"Found {len(game_files)} game files")
    
    all_states = []
    all_actions = []
    
    # Load and combine all game data
    for file_path in game_files:
        try:
            states, actions = load_game_data(file_path)
            if len(states) == len(actions): # Verify data consistency
                all_states.extend(states)
                all_actions.extend(actions)
            else:
                print(f"Skipping {file_path} - inconsistent data")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    print(f"Total number of moves: {len(all_states)}")
    
    if len(all_states) == 0:
        print("No valid training data found!")
        return
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = preprocess_data(all_states, all_actions)
    X = X.reshape(-1, 4, 4, 1) # Reshape for CNN
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Calculate split size based on actual data size
    validation_split = GEN_CONFIG['model']['validation_split']
    split_size = int(len(X) * (1 - validation_split))
    
    # Generate indices only up to data size
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[:split_size], indices[split_size:]
    
    # Split the data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create and train model
    print("Creating model...")
    model = create_model()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=GEN_CONFIG['model']['early_stopping_patience'],
            restore_best_weights=True
        )
    ]
    
    # ModelCheckpoint only if we want to save intermediate models
    if GEN_CONFIG['model']['save_frequency'] != 'last_only':
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                'models/2048_model_{epoch:02d}_{val_accuracy:.3f}.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        )
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=GEN_CONFIG['model']['epochs'],
        batch_size=GEN_CONFIG['model']['batch_size'],
        callbacks=callbacks
    )
    
    model.save('models/2048_model_final.h5')
    
    print("\nTraining completed!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")

if __name__ == '__main__':
    main()