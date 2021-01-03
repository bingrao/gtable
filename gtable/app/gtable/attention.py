import torch.nn.functional as F
import torch.nn as nn
import copy
import math
from torch.nn import Linear, Module, ReLU, LayerNorm, Dropout
import torch


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# [batch_size, h, n_col, d_k], torch.Size([500, 2, 24, 16])
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)

    # [batch_size, h, n_col, n_col], torch.Size([500, 2, 24, 24])
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # torch.Size([500, 2, 24, 24])

    if dropout is not None:
        p_attn = dropout(p_attn)

    # torch.matmul(p_attn, value): [batch_size, h, n_col, d_k], torch.Size([500, 2, 24, 16])
    # p_attn: [batch_size, h, n_col, n_col], torch.Size([500, 2, 24, 24])
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_col, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.n_col = n_col  # nums columns of input
        self.h = h  # nums of head
        self.d_model = d_model  # dimension size of input and output model
        self.d_k = math.ceil(self.d_model / self.n_col / self.h)
        self.new_d_model = self.d_k * self.n_col * self.h

        # [query_linear, key_linear, value_linear, final_linear]
        self.linears = clones(nn.Linear(self.new_d_model, self.new_d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def shape(self, x):
        # x: [batch_size, d_model], torch.Size([500, 743])
        # batch_size
        nbatches = x.size(0)

        # tgt: [batch_size, new_d_model], torch.Size([500, 768])
        tgt = torch.zeros(nbatches, self.new_d_model).to(x.device)
        tgt[:, :self.d_model] = x
        return tgt

    def unshape(self, x):
        return x[:, :self.d_model]

    def forward(self, _input, mask=None):
        # _input: [batch_size, d_model], torch.Size([500, 743])
        # batch_size, 500
        nbatches = _input.size(0)

        # query: [batch_size, new_d_model], torch.Size([500, 768])
        query = self.shape(_input)
        key = self.shape(_input)
        value = self.shape(_input)

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do individual linear projection on query, key, value
        # in batch from d_model => h x d_k,
        # query = linear_fn(query)
        # key = linear_fn(key)
        # value = linear_fn(value)
        # shape: torch.Size([batch_size, h, n_col, d_k]), torch.Size([500, 2, 24, 16])
        query, key, value = [ln(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for ln, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: [batch_size, h, n_col, d_k], torch.Size([500, 2, 24, 16])
        # self.attn: [batch_size, h, n_col, n_col], torch.Size([500, 2, 24, 24])
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # x: [batch_size, new_d_model], torch.Size([500, 768])
        x = x.transpose(1, 2).contiguous().view(nbatches, self.new_d_model)

        # output: [batch_size, d_model], e.g. torch.Size([500, 743])
        output = self.unshape(self.linears[-1](x))
        return output


class PositionwiseFeedForward(Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        input_dim (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
        output_dim (int): the size of input for the first-layer of the FFN.
    """

    def __init__(self, input_dim, d_ff, output_dim, dropout=0.1, residual=False):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = LayerNorm(input_dim, eps=1e-6)
        self.w_1 = Linear(input_dim, d_ff)
        self.w_2 = Linear(d_ff, output_dim)
        self.dropout_1 = Dropout(dropout)
        self.relu = ReLU()
        self.dropout_2 = Dropout(dropout)

        self.residual = True if (input_dim == output_dim) and residual else False

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))

        if self.residual:
            return output + x
        else:
            return output
