import math
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# ┏━━━━━━━━━━ To avoid Warning ━━━━━━━━━━┓
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,
                 patience: int = 7,
                 verbose: bool = False,
                 delta: float = 0,
                 path: str = "checkpoint.pt",
                 trace_func=print,
                 just_count: bool = False,
                 sharp_increase_factor: float = 10):
        """
        Args:
            patience              (int     ): How long to wait after last time validation loss improved.
            verbose               (bool    ): If True, prints a message for each validation loss improvement.
            delta                 (float   ): Minimum change in the monitored quantity to qualify as an improvement.
            path                  (str     ): Path for the checkpoint to be saved to.
            trace_func            (function): Trace print function.
            just_count            (bool    ): Whether to just count the minimum loss or actually save the model each time the loss is decreased.
            sharp_increase_factor (int     ): If each new loss is higher than the best loss times this number then there's a sharp loss increase and we're stopping
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.just_count = just_count
        self.sharp_increase_factor = sharp_increase_factor

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # ┏━━━━━━━━━━ Not usually triggered unless loss explodes by a huge factor  ━━━━━━━━━━┓
        elif score < self.best_score * self.sharp_increase_factor:
            self.trace_func("EarlyStopping: drastic loss increase → stopping")
            self.early_stop = True
        
        # ┏━━━━━━━━━━ No meaningful improvement → increment patience ━━━━━━━━━━┓
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        # ┏━━━━━━━━━━ Significant improvement ━━━━━━━━━━┓
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if not self.just_count:
            if self.verbose:
                self.trace_func(f"Validation loss: {self.val_loss_min:.6f} → {val_loss:.6f}.  Saving model …")
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class RevIN(nn.Module):
    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True):
        """
        Reversible Instance Norm over the time dimension: https://openreview.net/pdf?id=cGDAkQo1C0p
        """
        super().__init__()
        self.eps = eps
        self.affine = affine
        
        # ┏━━━━━━━━━━ Learnable Weight & Bias ━━━━━━━━━━┓
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))  # (C,)
            self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str = "norm"):
        if mode != "norm":
            raise NotImplementedError("RevIN denorm not required for classification.")
        dim2reduce = (-1,)                        

        # ┏━━━━━━━━━━ Instance Normalization ━━━━━━━━━━┓
        mean  = x.mean(dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - mean) / stdev

        if self.affine:
            w = self.weight.view(1, -1, 1)
            b = self.bias.view(1, -1, 1)
            x = x * w + b
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, matching the Transformer paper.
    Injects information about token position in the sequence.
    max_length (= num_tokens) = int{ (seq_len - kernel_size) / stride } + 1
    """
    def __init__(self, d_model, max_len=12):
        super().__init__()
        # ┏━━━━━━━━━━ Create constant 'pe' matrix with values dependent on position and i ━━━━━━━━━━┓
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # ┏━━━━━━━━━━ Even indices ━━━━━━━━━━┓
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # ┏━━━━━━━━━━ Odd indices ━━━━━━━━━━┓
        pe[:, 1::2] = torch.cos(position * div_term)   
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class LearnablePosEnc(nn.Module):
    """
    Classic BERT-style position embedding: one learned vector per position.
    We copy the weights from the deterministic sinusoid so the optimiser
    starts from a sensible point instead of random noise.

    """
    def __init__(self, d_model: int, max_len: int, p_dropout: float = 0.1):

        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.p = p_dropout

        # ┏━━━━━━━━━━ Will raise if context_len > max_len ━━━━━━━━━━┓
        self.register_buffer("_max_len", torch.tensor(max_len, dtype=torch.long))

        # ┏━━━━━━━━━━ Helpful for weight decay: initialise close to sinusoid ━━━━━━━━━━┓
        with torch.no_grad():
            sinusoid = PositionalEncoding(d_model, max_len).pe.squeeze(0)
            self.pe.weight.copy_(sinusoid)


    def forward(self, x):
        L = x.size(1)
        idx = torch.arange(x.size(1), device=x.device)
        pos_vec = self.pe(idx)
        pos_vec = F.dropout(pos_vec, p=self.p, training=self.training)
        return x + pos_vec

class CNNEncoder(nn.Module):
    """
    1D CNN encoder with optional SAME padding.
    - Supports multiple convolutional layers in sequence
    - Always starts with in_channels=1
    - Learnable positional encoding applied after the final conv
    """
    def __init__(self,
                 context_len: int,
                 embed_dim:   [int], # list of out_channels
                 kernel_size: [int], # list of kernel sizes
                 stride:      [int], # list of strides
                 p_pos_drop:  float,
                 padding:     bool,
                 in_channels: int = 1):
        super().__init__()

        # ┏━━━━━━━━━━ Sanity Check ━━━━━━━━━━┓
        assert len(embed_dim) == len(kernel_size) == len(stride), \
            "Error: embed_dim, kernel_size and stride must have the same length "
        
        self.use_padding = padding
        self.context_len = context_len

        # ┏━━━━━━━━━━ Build a Conv block for each (embed, kernel, stride) and in_channels is the number of features ━━━━━━━━━━┓
        convs: list[nn.Module] = []
        in_ch   = in_channels
        current_len = context_len

        for out_ch, kernel_size, stride in zip(embed_dim, kernel_size, stride):
            if self.use_padding:
                # ┏━━━━━━━━━━ Compute number of output tokens: ceil(context_len / stride) ━━━━━━━━━━┓
                out_tokens = math.ceil(current_len / stride)
                # ┏━━━━━━━━━━ Compute total padding needed to produce that many tokens ━━━━━━━━━━┓
                pad_needed = max(0, (out_tokens - 1) * stride + kernel_size - current_len)
                pad_left = pad_needed // 2
                pad_right = pad_needed - pad_left
                convs.append(nn.Sequential(
                    nn.ConstantPad1d((pad_left, pad_right), 0),
                    nn.Conv1d(in_ch, 
                              out_ch, 
                              kernel_size = kernel_size, 
                              stride = stride, 
                              padding = 0)))

                n_tokens = out_tokens
            else:
                # ┏━━━━━━━━━━ No padding: classic “valid” convolution ━━━━━━━━━━┓
                convs.append(nn.Conv1d(in_ch, 
                                       out_ch, 
                                       kernel_size = kernel_size, 
                                       stride = stride, 
                                       padding = 0))
                n_tokens = (current_len - kernel_size) // stride + 1

            in_ch = out_ch  # next layer’s in_channels
            current_len = n_tokens

        # ┏━━━━━━━━━━ Multiple Conv1d always internal-padding=0; we pad manually if requested ━━━━━━━━━━┓
        self.convs   = nn.ModuleList(convs)
        
        # ┏━━━━━━━━━━ Learnable positions for exactly n_tokens ━━━━━━━━━━┓
        self.pos_enc = LearnablePosEnc(embed_dim[-1], max_len = n_tokens, p_dropout = p_pos_drop)

        # ┏━━━━━━━━━━ Positional Encoding for exactly n_tokens ━━━━━━━━━━┓
        #self.pos_enc = PositionalEncoding(embed_dim[-1], max_len = n_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, context_len)
        Returns:
            (batch, n_tokens, embed_dim)
        """
        for layer in self.convs:
            x = layer(x)                # (batch, out_ch_i, length_i)

        x = x.transpose(1, 2)           # (batch, length, embed_dims[-1])
        return self.pos_enc(x)          # still (batch, length, embed_dims[-1])



class TransformerEncoder(nn.Module):
    """
    Transformer encoder adapted for time-series token sequences.
    - embed_dim: dimensionality of input tokens
    - num_heads: number of self-attention heads
    - dim_ff: hidden dimension of the feed-forward network
    - num_layers: number of Transformer encoder layers (depth)
    - dropout: dropout rate for attention and feed-forward
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dim_ff,
                 num_layers,
                 dropout,
                 activation):

        super().__init__()

        # ┏━━━━━━━━━━ Activation Function ━━━━━━━━━━┓
        act_arg = activation.lower()
        if act_arg in {"gelu", "relu"}:
            act_spec = act_arg
        elif act_arg == "silu":
            act_spec = nn.SiLU()
        elif act_arg == "mish":
            act_spec = nn.Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ┏━━━━━━━━━━ Encoder Layer ━━━━━━━━━━┓
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            activation      = act_spec,
            batch_first     = False,
            norm_first      = True,
        )

        # ┏━━━━━━━━━━ Stack the specified number of layers  ━━━━━━━━━━┓
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            out: Tensor of same shape (batch_size, seq_len, embed_dim)
        """
        # ┏━━━━━━━━━━ Permute for PyTorch Transformer: (seq_len, batch_size, embed_dim)  ━━━━━━━━━━┓
        x = x.permute(1,0,2)       # (seq_len, batch, embed_dim)
        
        # ┏━━━━━━━━━━ Apply transformer encoder  ━━━━━━━━━━┓
        try:
            out = self.transformer(x)
        except RuntimeError as err:
            raise RuntimeError(
                f"TransformerEncoder failed for input shape {tuple(x.shape)} "
                f"(seq_len, batch, embed_dim) with error: {err}"
            ) from err
        
        # ┏━━━━━━━━━━ Permute back to (batch_size, seq_len, embed_dim)  ━━━━━━━━━━┓
        return out.permute(1,0,2)   # (batch, seq_len, embed_dim)


class AttentionPooling(nn.Module):
    """
    Self-attention pooling over time: x (B, T, D) → (B, D)
    """
    def __init__(self, dim: int):
        super().__init__()
        # ┏━━━━━━━━━━ Learnable Query Vector ━━━━━━━━━━┓
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ┏━━━━━━━━━━ Scores: (B, T) and x: (B, T, D) ━━━━━━━━━━┓
        scores = x @ self.query

        # ┏━━━━━━━━━━ Alpha: (B, T, 1) ━━━━━━━━━━┓
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        
        # ┏━━━━━━━━━━ Weighted sum: (B, D) ━━━━━━━━━━┓
        return (x * alpha).sum(dim=1)


class MeanMaxPooling(nn.Module):
    """
    Concatenate mean and max-pooling over time: x (B, T, D) → (B, 2D)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ┏━━━━━━━━━━ x: (B, T, D) ━━━━━━━━━━┓
        μ = x.mean(dim=1)                # (B, D)
        m = x.max(dim=1).values          # (B, D)
        return torch.cat([μ, m], dim=1)  # (B, 2D)


class ClassificationHead(nn.Module):
    def __init__(self,
                 embed_dim:   int,
                 hidden_dim:  int,
                 num_classes: int,
                 dropout:     float,
                 activation:  str,
                 pooling:     str):

        super().__init__()

        # ┏━━━━━━━━━━ Pooling ━━━━━━━━━━┓
        p = pooling.lower()
        if p == "attention":
            self.pool = AttentionPooling(embed_dim)
            mlp_in = embed_dim
        elif p == "meanmax":
            self.pool = MeanMaxPooling()
            mlp_in = 2 * embed_dim
        else:
            raise ValueError(f"Unknown pooling '{pooling}'")


        # ┏━━━━━━━━━━ Linear Layer ━━━━━━━━━━┓
        self.fc1     = nn.Linear(mlp_in, hidden_dim)
        
        # ┏━━━━━━━━━━ Activation Function ━━━━━━━━━━┓
        self.act     = {"gelu":    nn.GELU(),
                        "relu":    nn.ReLU(),
                        "lrelu":   nn.LeakyReLU(negative_slope=0.01),
                        "silu":    nn.SiLU(),
                        "mish":    nn.Mish()}[activation.lower()]
        
        # ┏━━━━━━━━━━ Dropout ━━━━━━━━━━┓
        self.dropout = nn.Dropout(dropout)
        
        # ┏━━━━━━━━━━ Linear Layer ━━━━━━━━━━┓
        self.fc2     = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        # ┏━━━━━━━━━━ Mean-Max or Attention Pooling ━━━━━━━━━━┓
        x = self.pool(x)
        
        # ┏━━━━━━━━━━ Linear Layer ━━━━━━━━━━┓
        x = self.fc1(x)

        # ┏━━━━━━━━━━ Activation Function ━━━━━━━━━━┓
        x = self.act(x)
        
        # ┏━━━━━━━━━━ Dropout ━━━━━━━━━━┓
        x = self.dropout(x)

        # ┏━━━━━━━━━━ Linear Layer ━━━━━━━━━━┓
        x = self.fc2(x) # (batch, num_classes)
   
        return x  


class CTTSModel(nn.Module):
    """
    Full CTTS: RevIN → CNNEncoder1D → TransformerEncoder → ClassificationHead.
    """
    def __init__(self,
                 cnn_embed_dim: [int],
                 cnn_kernel:    [int],
                 cnn_stride:    [int],
                 p_pos_drop:    float,
                 nb_features:   int,
                 
                 trans_heads:   int,
                 trans_layers:  int,
                 trans_ff:      int,
                 trans_dropout: float,
                 trans_activ:   str,
                 
                 mlp_hidden:    int,
                 mlp_dropout:   float,
                 mlp_activ:     str,
                 mlp_pooling:   str,

                 num_classes:   int,
                 padding:       bool,
                 context_len:   int):

        super().__init__()

        # ┏━━━━━━━━━━ 2 Quick Sanity Checks - fail fast if YAML misses a key ━━━━━━━━━━┓
        for name, val in locals().items():
            if name != "self" and val is None:
                raise ValueError(f"CTTSModel: parameter '{name}' is None")
        assert len(cnn_embed_dim) == len(cnn_kernel) == len(cnn_stride), ("Error: cnn_embed_dim, cnn_kernel and cnn_stride must all be the same length")

        # ┏━━━━━━━━━━ Number of channels after the final convolution ━━━━━━━━━━┓
        last_embed = cnn_embed_dim[-1]

        # ┏━━━━━━━━━━ RevIn ━━━━━━━━━━┓
        self.revIn = RevIN(num_features = nb_features)

        # ┏━━━━━━━━━━ CNN ━━━━━━━━━━┓
        self.cnn   = CNNEncoder(context_len   = context_len,
                                  embed_dim   = cnn_embed_dim,
                                  kernel_size = cnn_kernel,
                                  stride      = cnn_stride,
                                  p_pos_drop  = p_pos_drop,
                                  padding     = padding,
                                  in_channels = nb_features)
        
        # ┏━━━━━━━━━━ Transformer Encoder ━━━━━━━━━━┓
        self.trans = TransformerEncoder(embed_dim  = last_embed,
                                        num_heads  = trans_heads,
                                        dim_ff     = trans_ff,
                                        num_layers = trans_layers,
                                        dropout    = trans_dropout,
                                        activation = trans_activ)

        # ┏━━━━━━━━━━ Classification MLP ━━━━━━━━━━┓                        
        self.head  = ClassificationHead(embed_dim   = last_embed,
                                        hidden_dim  = mlp_hidden,
                                        num_classes = num_classes,
                                        dropout     = mlp_dropout,
                                        activation  = mlp_activ,
                                        pooling     = mlp_pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)
        x       = self.revIn(x, mode='norm') # (batch, num_tokens, embed_dim)
        tokens  = self.cnn(x)                # same shape
        encoded = self.trans(tokens)         # (batch, num_classes)
        return self.head(encoded)
        
