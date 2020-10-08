from torch import nn
from labml_helpers.module import Module
from labml_nn.lstm import LSTM

class LSTM_Model(Module):
    
    def __init__(self, *, n_tokens, embedding_size, hidden_size, n_layers):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers)
        self.fc = nn.Linear(hidden_size, n_tokens)

    def __call__(self, x, h0=None, c0=None):
        x = self.embedding(x)
        state = (h0, c0) if h0 is not None else None
        out, (hn, cn) = self.lstm(x, state)
        logits = self.fc(out)
        return logits, (hn, cn)