from dataclasses import dataclass
import os

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True

print(os.path.dirname(__file__))
