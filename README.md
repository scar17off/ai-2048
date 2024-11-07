# 2048 Game with AI Self-Learning

A Python implementation of the popular 2048 game featuring an AI agent that learns to play through self-training using TensorFlow.

## Features

- Classic 2048 game implementation
- AI agent powered by TensorFlow
- Self-learning capabilities through reinforcement learning
- GPU acceleration support
- Real-time visualization of game states
- Performance logging and monitoring
- Training progress visualization using Matplotlib

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pygame
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/scar17off/ai-2048.git
cd 2048-ai-self-learning
```

2. Install dependencies:
```bash
pip install tensorflow numpy pygame matplotlib
```

## Project Structure

- `self_learn.py`: Main self-learning AI implementation
- `game_logic.py`: 2048 game mechanics
- `train_model.py`: Model training orchestration
- `ai_play.py`: AI gameplay execution
- `colors.py`: Game interface color definitions

## Usage

1. Generate training games:
```bash
python generate_training_games.py
```
Or download pre-generated games (dataset) in releases.

2. Train the AI model:
```bash
python train_model.py
```
Or download pre-trained model in releases.

3. Watch the AI play:
```bash
python ai_play.py
```

## Technical Details

### Running on GPU

The project automatically configures GPU settings for optimal performance. It enables memory growth and selects the primary GPU for processing, as shown in `self_learn.py`. 

To run the project on GPU:

1. Install TensorFlow GPU version:
```bash
pip install tensorflow-gpu
```

2. Install CUDA (https://developer.nvidia.com/cuda-downloads) version 12.0 or later.

3. Run the project as usual.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE file](LICENSE.md) for details.

## Acknowledgments

- Original 2048 game by Gabriele Cirulli
- TensorFlow team for the excellent machine learning framework