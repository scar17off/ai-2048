from abc import ABC, abstractmethod
from game_logic import Game2048

class BasePlayStyle(ABC):
    def __init__(self, game_config, gen_config):
        self.game_config = game_config
        self.gen_config = gen_config

    @abstractmethod
    def generate_move(self, game: Game2048) -> str:
        """Generate the next move based on the current game state"""
        pass 	