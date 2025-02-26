import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Implements a self-attention mechanism to capture dependencies in sequential data.

    Args:
        hidden_size (int): Size of the hidden representation.
        attention_size (int): Size of the attention layer.
        n_attention_heads (int): Number of attention heads.
    """

    def __init__(self, hidden_size, attention_size, n_attention_heads):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads

        # Linear layers for attention mechanism
        self.W1 = nn.Linear(hidden_size, attention_size, bias=True)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=True)

    def forward(self, hidden):
        """
        Forward pass of self-attention mechanism.

        Args:
            hidden (Tensor): Input tensor of shape (batch_size, sentence_length, hidden_size).

        Returns:
            M (Tensor): Attention-weighted hidden representations.
            A (Tensor): Attention weight matrix.
        """

        # Apply non-linearity to projected hidden states
        x = torch.tanh(self.W1(hidden))

        # Compute attention scores and normalize with softmax over sentence length
        x = F.softmax(self.W2(x), dim=1)

        # Transpose to obtain attention matrix
        A = x.transpose(1, 2)

        # Compute attention-weighted hidden representations
        M = A @ hidden

        return M, A


class GLEAD(nn.Module):
    """
    Implements GLEAD network.

    Args:
        attention_size (int): Size of the attention layer.
        n_attention_heads (int): Number of attention heads.
        hidden_size (int): Size of the hidden representation.
    """

    def __init__(self, attention_size, n_attention_heads, hidden_size, logkeys, device):
        super().__init__()

        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.hidden_size = hidden_size
        self.alpha = 1.0  # Scaling factor for softmax normalization

        # Embedding layer for event sequences
        self.embedding = nn.Embedding(num_embeddings=len(logkeys), embedding_dim=hidden_size)

        # Self-attention module
        self.self_attention = SelfAttention(
            hidden_size=self.hidden_size,
            attention_size=attention_size,
            n_attention_heads=n_attention_heads
        )

        # Context vectors for normal and anomalous samples, randomly initialized
        self.c_n = nn.Parameter((torch.rand(1, n_attention_heads, self.hidden_size) - 0.5) * 2)
        self.c_a = nn.Parameter((torch.rand(1, n_attention_heads, self.hidden_size) - 0.5) * 2)

        # Cosine similarity function
        self.cosine_dist = nn.CosineSimilarity(dim=2)

        self.device = device

    def forward(self, x, sequence_label, semi, batch_size, hidden_size):
        """
        Forward pass of GLEAD.

        Args:
            x (Tensor): Input event sequence indices.
            sequence_label (Tensor): Ground truth labels (normal/anomalous).
            semi (Tensor): Semi-supervised labels (0 for labeled, 1 for unlabeled).
            batch_size (int): Batch size.
            hidden_size (int): Hidden representation size.

        Returns:
            triplet_loss (Tensor): Triplet loss for anomaly detection.
            dists (tuple): Distance metrics used in loss computation.
            context_weights (tuple): Context weights from softmax normalization.
            M (Tensor): Attention-weighted hidden representations.
            A (Tensor): Attention weight matrix.
        """

        # Obtain hidden representations from embeddings
        hidden = self.embedding(x.to(self.device))

        # Apply self-attention mechanism
        M, A = self.self_attention(hidden)

        # Separate data into labeled and unlabeled categories
        M_u = M[semi == 1]  # Unlabeled data
        M_n = M[(semi == 0) & (sequence_label == 0)]  # Labeled normal data
        M_a = M[(semi == 0) & (sequence_label == 1)]  # Labeled anomalous data

        # Combine normal and unlabeled data
        M_n = torch.cat((M_u, M_n), dim=0)

        # Expand context vectors to match batch size
        c_n_n = torch.repeat_interleave(self.c_n, M_n.size(0), dim=0)
        c_a_n = torch.repeat_interleave(self.c_a, M_n.size(0), dim=0)
        c_a_a = torch.repeat_interleave(self.c_a, M_a.size(0), dim=0)
        c_n_a = torch.repeat_interleave(self.c_n, M_a.size(0), dim=0)

        # Compute cosine distances between representations and context vectors
        distnn = 0.5 * (1 - self.cosine_dist(M_n, c_n_n))
        distna = 0.5 * (1 - self.cosine_dist(M_n, c_a_n))
        distaa = 0.5 * (1 - self.cosine_dist(M_a, c_a_a))
        distan = 0.5 * (1 - self.cosine_dist(M_a, c_n_a))

        # Compute context weights using softmax normalization
        context_weights_nn = torch.softmax(-self.alpha * distnn, dim=1)
        context_weights_na = torch.softmax(self.alpha * distna, dim=1)
        context_weights_aa = torch.softmax(-self.alpha * distaa, dim=1)
        context_weights_an = torch.softmax(self.alpha * distan, dim=1)

        # Prepare output tuples
        dists = (distnn, distna, distaa, distan)
        context_weights = (context_weights_nn, context_weights_na, context_weights_aa, context_weights_an)

        # Compute triplet loss for anomaly detection
        triplet_loss1 = torch.sum(distnn * context_weights_nn, dim=1) - torch.sum(distna * context_weights_na,
                                                                                  dim=1) + 1
        triplet_loss2 = torch.sum(distaa * context_weights_aa, dim=1) - torch.sum(distan * context_weights_an,
                                                                                  dim=1) + 1

        # Apply ReLU activation to enforce non-negative loss and compute mean loss
        triplet_loss = torch.sum(torch.relu(triplet_loss1)) / (triplet_loss1.size(0) + 1) + \
                       torch.sum(torch.relu(triplet_loss2)) / (triplet_loss2.size(0) + 1)

        return triplet_loss, dists, context_weights, M, A
