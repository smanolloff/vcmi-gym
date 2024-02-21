from dataclasses import dataclass

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
