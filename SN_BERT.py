



# single nucleotide Constraint BERT with Pytorch SDPA




from typing import cast, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, filename, sequence_length=1024, stride=512):
        with open(filename, 'r') as file:
            self.text = file.read()
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self):
        return (len(self.text) - self.sequence_length) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.sequence_length
        sample = self.text[start:end]
        return sample


class mha_self(nn.Module):
    """
    An implementation of multi-head (self) attention using
    torch.nn.functional.scaled_dot_product_attention.
    This allows flash attention v2 if you wrap forward call in:
    `with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):`
    """
    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 qkv_bias: bool = False):
        super(mha_self, self).__init__()

        if dim_out % n_heads != 0:
            raise ValueError("dim_in must be divisible by n_heads")

        self.dim_in: int = dim_in
        self.dim_out: int = dim_out
        self.n_heads: int = n_heads
        self.dropout: float = dropout
        self.head_dim: int = dim_out // n_heads

        self.qkv = nn.Linear(dim_in, 3 * dim_out, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            The input tensor to apply multi-head attention to. Must be
            3-dimensional with shape (batch_size, num_tokens, embed_dim).
        key_padding_mask: torch.Tensor
            Optional boolean tensor of shape (batch_size, num_tokens) where
            True values indicate padding tokens. These tokens will be ignored
            in the attention calculation.
        attn_mask: torch.Tensor
            Optional boolean tensor of shape (batch_size, num_tokens, num_tokens)
            where True values indicate that the corresponding token should be
            ignored in the attention calculation.
        """
        batch_size = x.size(0)
        num_tokens = x.size(1)

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, -1, 3, self.n_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3x (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)

        # attention mask per head
        amask: Optional[Tensor] = None
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(1, self.n_heads, 1, 1)
            amask = attn_mask

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            key_padding_mask = key_padding_mask.repeat(1, self.n_heads, num_tokens, 1)
            if amask is None:
                amask = key_padding_mask
            else:
                amask = amask | key_padding_mask

        use_dropout = 0. if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=amask, dropout_p=use_dropout,
        )

        # Combine heads with `view`, self.dim_out = self.n_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(
            batch_size, -1, self.dim_out
        )

        # out projection
        context_vec = self.proj(context_vec)

        return context_vec


class TfBlock(nn.Module):
    """
    A single transformer block consisting of multi-head attention and
    a multi-layer perceptron.

    Parameters
    ----------
    dim: int
        The dimension of each input token in the sequence. In other words,
        the number of features in the input.
    n_heads: int
        The number of heads in the multi-head attention mechanism.
        Must be a factor of `dim`.
    dropout: float
        The dropout rate to apply to the attention output weights.
    bias: bool
        Whether to include bias in the linear layers.
    dim_feedforward: int
        The dimension of the hidden layer in the multi-layer perceptron.
    batch_first: bool
        Whether the input and output tensors are batch-first. If true,
        input and output are (batch, seq, feature)

    """
    def __init__(self,
                 dim: int = 256,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 attn_weight_dropout: float = 0.0,
                 qkv_bias: bool = False,
                 dim_feedforward: int = 1024):

        super(TfBlock, self).__init__()

        # check if dim is divisible by n_heads
        if dim % n_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by n_heads={n_heads}"
            )

        self.dim: int = dim
        self.n_heads: int = n_heads
        self.attn: mha_self = mha_self(
            dim_in=dim,
            dim_out=dim,
            n_heads=n_heads,
            dropout=attn_weight_dropout,
            qkv_bias=qkv_bias
        )
        self.dropout1: nn.Dropout = nn.Dropout(p=dropout)
        self.dropout2: nn.Dropout = nn.Dropout(p=dropout)
        self.ln1: nn.LayerNorm = nn.LayerNorm(dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(dim)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GeLU(), #nn.ReLU(),
            nn.Linear(dim_feedforward, dim)
        )

# old encoder block style
    def forward(self,
                    x: Tensor,
                    key_padding_mask: Optional[Tensor] = None,
                    attn_mask: Optional[Tensor] = None) -> torch.Tensor:
        """
        Run forward pass without activation checkpoints.
        """
        attn_output = self.attn(
            x=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
        x = self.ln1(x + self.dropout1(attn_output))
        mlp_output = self.mlp(x)
        return self.ln2(x + self.dropout2(mlp_output))

# new encoder block style
    def new_forward(self,
                    x: Tensor,
                    src_key_padding_mask: Optional[Tensor] = None,
                    attn_mask: Optional[Tensor] = None) -> torch.Tensor:
        """
        Run forward pass without activation checkpoints.
        """
        # apply layer norm first
        LN_SAs = self.ln1(x)
        
        # apply MHA
        attn_output = self.attn(
            x=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )

        # add MHA and residuals
        x = self.ln2(x + self.dropout1(attn_output))

        # feed forward network
        mlp_output = self.mlp(x)
        return (x + self.dropout2(mlp_output))


# still need the complete BERT model

# add the Roberta LM head
class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, alphabet_size):
        super().__init__()
        # add another layer norm in the front, our Transformer Blocks are LN first.
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.LL1 = nn.Linear(embed_dim, embed_dim)
        self.gelu = nn.GELU() 
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.LL2 = nn.Linear(embed_dim, alphabet_size)

    def forward(self, x):
        x = self.layer_norm1(x)
        # get embeddings after layer norm
        embeddings = x
        x = self.LL1(x)
        x = self.gelu(x)   
        x = self.layer_norm2(x)
        # compute logits
        logits = self.LL2(x)
        return {"embeddings": embeddings, "logits": logits}

# whole model
class ConstraintBertModel(nn.Module):
    def __init__(self, tokenizer, encoder_layers = 10, 
                 embedding_dim = 1024, ffn_embedding_dim = 4096,  
                #embedding_dim = 8192, ffn_embedding_dim = 24576,
                 #embedding_dim = 2048, ffn_embedding_dim = 6144,
                 #head_num = 64,
                 head_num = 16, 
                 dropout=0.1):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.head_num = head_num
        self.dropout = dropout
        self.args = dict()
        self.args["token_dropout"] = False

        # adjust this code to use a Tokenizers tokenizer from Hugging Face
        self.alphabet_size = (tokenizer.get_vocab_size())
        self.padding_idx = tokenizer.token_to_id('<PAD>')
        self.mask_idx = tokenizer.token_to_id('<MASK>')
        self.cls_token_id = tokenizer.token_to_id('<CLS>')
        self.eos_token_id = tokenizer.token_to_id('<EOS>')
        # self.prepend_bos = alphabet.prepend_bos
        # self.append_eos = alphabet.append_eos
        self._init_submodules()
    
    def _init_submodules(self):
        # embed the tokens
        self.embed_tokens = nn.Embedding(
                self.alphabet_size, self.embedding_dim, padding_idx=self.padding_idx
            )
        self.embed_scale = math.sqrt(self.embedding_dim)
        # add positional encoding
        self.embed_positions = SinusoidalPositionalEmbedding(self.embedding_dim, self.padding_idx)
        # add nn.TransformerEncoderLayer to fsdp and activation checkpointing
        self.whole_encoder = nn.Sequential(*[
            TfBlock(
                dim=self.embedding_dim,
                n_heads=self.head_num,
                dropout=self.dropout,
                attn_weight_dropout=0.0, # what should this number be?
                qkv_bias=False,
                dim_feedforward=self.ffn_embedding_dim
            ) for _ in range(self.encoder_layers)
        ])
        # encoder_block = nn.TransformerEncoderLayer(d_model = self.embedding_dim, 
        #                                          nhead = self.head_num, 
        #                                          dim_feedforward = self.ffn_embedding_dim,
        #                                          activation ="gelu",
        #                                          batch_first = True, # expects batch dim first
        #                                          norm_first = True, # more stable
        #                                          dropout = self.dropout)
        # # don't we need the q and k heads exposed for rotary encoding?
        # self.whole_encoder = nn.TransformerEncoder(encoder_block, 
        #                                      num_layers = self.encoder_layers)
        # need LM head for BERT
        # roberta lm head is layer norm -> LL -> activation -> layer norm -> LL 
        self.lm_head = RobertaLMHead(
            embed_dim=self.embedding_dim,
            alphabet_size=self.alphabet_size,
        )
    def forward(self, tokens):
        # get the padding mask
        padding_mask = tokens.eq(self.padding_idx)  # B, T
        #print("padding mask", padding_mask.shape)
        #print("device", tokens.device)
        # compute the embeddings and apply RoBERTa's mask scaling factor
        x = self.embed_scale * self.embed_tokens(tokens)
        #print("token embeddings", x)
        # add add token dropout/ BERT masking
        if getattr(self.args, "token_dropout", False):
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        #print("x after mask", x)
        # add positional encodings
        x = x + self.embed_positions(tokens)
        #print("x after positional embedding", x)
        # needed for some reason? -- try with the built in encoder layers and see if any issues
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        #print("x after padding mask unsqueeze", x)
        if not padding_mask.any():
            padding_mask = None

        # x should be shape(B, seq_len, embedding_dim)
        # padding mask should be (B, seq_len)
        # put it through the Transformer tower, pass the padding mask
        x = self.whole_encoder(x, src_key_padding_mask=padding_mask)
        #print("x after encoding", x)
        # apply the LM head
        logit_embed_dict = self.lm_head(x)
        # print("logits", logit_embed_dict["logits"].shape)
        # print("embeddings", logit_embed_dict["embeddings"].shape)
        # logit_embed_dict = {"logits": x, "embeddings": hidden_representations}
        # get the logit and embeddings
        return logit_embed_dict

# need an arg parser   
def get_args(parser):
    parser.add_argument("--chunk-dir", required = False, help = "dir where chr chunks are")# directory")
    parser.add_argument("--epoch", default = 2, type = int, help = "Number of epochs")
    # need more args
    return parser

# we should have everything we need now. Now we just need to debug to see that it can train


def main():

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Customize training
    trainer = trainers.BpeTrainer(vocab_size=256, min_frequency=1, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    tokenizer.train(files=["../phastcons_smoothed/chr1_smoothed_str.txt"], trainer=trainer)

    # Now you can encode text
    output = tokenizer.encode("Hello, world!")
    print(output.tokens)
    raise Error

    # Parse the arguments
    args = get_args(ArgumentParser()).parse_args()
    criterion = nn.CrossEntropyLoss()
    
    epoch_num = 1
    use_wandb = False
    multi_GPU = False
    WORLD_SIZE = torch.cuda.device_count()

    #train_path, val_path = "train_samples", "val_samples"
    # Create a Dataset
    dataset = TextDataset('large_text_file.txt')

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32)

    # Now you can iterate over the DataLoader
    for batch in dataloader:
        print(batch)
    # need the single nucleotide tokenizer here

    tokenizer = Tokenizer.from_file("sn_tokenizer.json") 
    #"transcript_tokenizer.json")#"chr1_tokenizer.json")
    print("tokenizer loaded")
    # you need to add all the special tokens to tokenizer
    special_tokens = ['<PAD>', '<MASK>', '<CLS>', '<EOS>']
    num_added_toks = tokenizer.add_tokens(special_tokens)

    print((tokenizer.get_vocab_size())) # 30,002 --> should be 4

    # you could still do flash attention with Rotary Encoding, you would just need to SDPA directly
    # that would expose the heads to do it 
    # need to init the model
    model = ConstraintBertModel(tokenizer)
    #print([param for param in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)# lr=5e-5)
    #print("hi", (optimizer))
    # may want a scheduler later
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    print(1)
    #
    if multi_GPU:
        print(2)
        # however the model doesn't seem to be training. No detectable movement in weights at all 
        mp.spawn(fsdp_main,
                args = (WORLD_SIZE, args.chunk_dir, 
                        epoch_num, criterion, model, optimizer, tokenizer),
                nprocs = WORLD_SIZE,
                join = True
                )
        #wandb.finish()
    else:
        if use_wandb:
            wandb.init(project="ConstraintBERT")

        single_GPU_main(train_path, val_path, epoch_num, model, optimizer, criterion, use_wandb, tokenizer)

        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
