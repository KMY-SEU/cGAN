import torch
import torch.nn as nn


class SSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(B, N, T, self.heads, self.head_dim)  # embed_size => heads×head_dim
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        query = query.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)  # (B, N, T, heads, head_dim)
        keys = self.keys(keys)  # (B, N, T, heads, head_dim)
        queries = self.queries(query)  # (B, N, T, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum('bqthd, bkthd -> bqkth', [queries, keys])  # self-attention
        # queries shape: (B, N, T, heads, heads_dim),
        # keys shape: (B, N, T, heads, heads_dim)
        # energy: (B, N, N, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=1)
        # attention shape: (N, N, T, heads)

        out = torch.einsum("bqkth, bkthd -> bqthd", [attention, values]).reshape(
            B, N, T, self.heads * self.head_dim
        )
        # attention shape: (B, N, N, T, heads)
        # values shape: (B, N, T, heads, heads_dim)
        # out after matrix multiply: (B, N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (B, N, T, embed_size)

        return out


class TSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        B, N, T, C = query.shape

        # Split the embedding into self.heads different pieces
        values = values.reshape(B, N, T, self.heads, self.head_dim)  # embed_size => heads×head_dim
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        query = query.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)  # (N, T, heads, head_dim)
        keys = self.keys(keys)  # (N, T, heads, head_dim)
        queries = self.queries(query)  # (N, T, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("bnqhd, bnkhd -> bnqkh", [queries, keys])  # self-attention
        # queries shape: (B, N, T, heads, heads_dim),
        # keys shape: (B, N, T, heads, heads_dim)
        # energy: (B, N, T, T, heads)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        # attention shape: (N, query_len, key_len, heads)

        out = torch.einsum("bnqkh, bnkhd -> bnqhd", [attention, values]).reshape(
            B, N, T, self.heads * self.head_dim
        )
        # attention shape: (N, T, T, heads)
        # values shape: (N, T, heads, heads_dim)
        # out after matrix multiply: (N, T, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, T, embed_size)

        return out


class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, dropout, forward_expansion):
        super(STransformer, self).__init__()
        self.devices = self.check_cuda()

        # Spatial Embedding
        self.D_S = nn.Parameter(adj, requires_grad=False)
        self.embed_linear = nn.Linear(adj.shape[0], embed_size)

        self.attention = SSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def check_cuda(self):
        # check cuda
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def forward(self, value):
        # Spatial Embedding
        B, N, T, C = value.shape
        D_S = self.embed_linear(self.D_S)
        D_S = D_S.expand(B, T, N, C)
        D_S = D_S.permute(0, 2, 1, 3)

        # Spatial Transformer
        value = value + D_S
        attention = self.attention(value, value, value)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + value))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        # Temporal embedding
        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding

        self.attention = TSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        B, N, T, C = value.shape

        D_T = self.temporal_embedding(torch.arange(0, T, device='cuda'))  # temporal embedding

        # temporal embedding
        value = value + D_T
        attention = self.attention(value, value, value)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + value))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, adj, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.merge = nn.Linear(3 * embed_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        # Add skip connection,run through normalization and finally dropout
        x1 = self.STransformer(value)
        x2 = self.TTransformer(value)

        out = self.merge(torch.cat([x1, x2, value], dim=3))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            forward_expansion,
            dropout,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout(x)

        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out)

        return out


class STTransformer_G(nn.Module):
    def __init__(
            self,
            adj,
            in_channels=1,
            embed_size=64,
            num_layers=3,
            heads=4,
            time_num=288,
            forward_expansion=4,
            dropout=0
    ):
        super(STTransformer_G, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            forward_expansion,
            dropout
        )

        # linear
        self.linear1 = nn.Linear(in_channels, embed_size)
        self.linear2 = nn.Linear(embed_size, in_channels)

    def forward(self, x):
        x = x.unsqueeze(dim=3)
        # [B, N, T, C]
        x = self.linear1(x)
        enc = self.encoder(x)
        out = self.linear2(enc)
        out = out.squeeze(dim=3)

        return out


class STTransformer_D(nn.Module):
    def __init__(
            self,
            adj,
            in_channels=1,
            nodes=66,
            features=144,
            embed_size=64,
            num_layers=1,
            heads=4,
            time_num=288,
            forward_expansion=4,
            dropout=0
    ):
        super(STTransformer_D, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            forward_expansion,
            dropout
        )

        # 1x1 convolution
        self.linear1 = nn.Linear(in_channels, embed_size)
        self.critic = nn.Sequential(
            nn.Linear(embed_size, in_channels),
            nn.Flatten(),
            nn.Linear(nodes * features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(dim=3)
        # [B, N, T, C]
        x = self.linear1(x)
        enc = self.encoder(x)
        out = self.critic(enc)

        return out
