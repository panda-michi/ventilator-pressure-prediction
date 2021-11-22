from dataclasses import dataclass

@dataclass
class bidLSTM_conf():
    input_dim: int = 50
    hidden0: int = 1024
    hidden1: int = 512
    hidden2: int = 256
    hidden3: int = 128

    dense: int = 128

@dataclass
class dlast_conf():
    input_dim : int = 23
