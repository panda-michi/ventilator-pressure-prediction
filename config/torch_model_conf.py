from dataclasses import dataclass

@dataclass
class simple_lstm():
    input_dim: int = 25
    lstm_dim: int = 256
    dense_dim: int = 256
    logit_dim: int = 256
    num_classes: int = 1

@dataclass
class embed_lstm():
    in_dim: int = 23
    sq_dim: int = 64
    hidden: int = 256
    dense: int = 50

@dataclass
class ts_lstm():
    in_dim: int = 5
    sq_dim: int = 64

    hidden0: int = 400
    hidden1: int = 300
    hidden2: int = 200
    hidden3: int = 100

    dense: int = 50

@dataclass
class dual_deep_lstm():
    in_dim: int = 23
    hidden: int = 512
    dense: int = 50