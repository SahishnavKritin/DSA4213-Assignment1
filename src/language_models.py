import torch
import torch.nn as nn


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100,
                 hidden_dim=128, num_layers=1, pad_id=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )

        self.rnn = nn.RNN(
            embedding_dim, hidden_dim,
            num_layers=num_layers, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100,
                 hidden_dim=128, num_layers=1, pad_id=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        return self.fc(output)


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100,
                 num_heads=4, num_layers=2, ff_dim=256,
                 max_seq_length=32, dropout=0.1, pad_id=0):
        super().__init__()

        self.max_seq_length = max_seq_length

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            max_seq_length, embedding_dim
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        seq_length = x.shape[1]

        if seq_length > self.max_seq_length:
            raise ValueError("Sequence exceeds context length")

        positions = torch.arange(
            seq_length, device=x.device
        )

        embedded = (
            self.token_embedding(x)
            + self.position_embedding(positions).unsqueeze(0)
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_length, seq_length,
                dtype=torch.bool, device=x.device
            ),
            diagonal=1
        )

        output = self.transformer(
            embedded, mask=causal_mask
        )

        return self.fc(output)