
# single nucleotide Constraint BERT with Pytorch SDPA
from torch.cuda.amp import autocast
from typing import cast, Tuple, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
import glob
import math
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

# add positional encodings
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False): # what does learned do?
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb



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

        self.qkv = nn.Linear(dim_in, 3 * dim_out, bias=qkv_bias, device=0)
        self.proj = nn.Linear(dim_out, dim_out, device = 0)

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
        self.ln1: nn.LayerNorm = nn.LayerNorm(dim, device=0)
        self.ln2: nn.LayerNorm = nn.LayerNorm(dim, device=0)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(dim, dim_feedforward, device=0),
            nn.GELU(), #nn.ReLU(),
            nn.Linear(dim_feedforward, dim, device =0)
        )

# new encoder block style
    def forward(self,
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
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask
        )

        # add MHA and residuals
        x = self.ln2(x + self.dropout1(attn_output))

        # feed forward network
        mlp_output = self.mlp(x)
        return (x + self.dropout2(mlp_output))

# old encoder block style
    def old_forward(self,
                    x: Tensor,
                    src_key_padding_mask: Optional[Tensor] = None,
                    attn_mask: Optional[Tensor] = None) -> torch.Tensor:
        """
        Run forward pass without activation checkpoints.
        """
        attn_output = self.attn(
            x=x,
            key_padding_mask=src_key_padding_mask,
            attn_mask=attn_mask
        )
        x = self.ln1(x + self.dropout1(attn_output))
        mlp_output = self.mlp(x)
        return self.ln2(x + self.dropout2(mlp_output))

# still need the complete BERT model
class CustomModule(nn.Module):
    def __init__(self, blocks):
        super(CustomModule, self).__init__()
        self.blocks = blocks

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        for block in self.blocks:
            x = block(x, src_key_padding_mask, attn_mask)
        return x

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

        self.whole_encoder = CustomModule(nn.Sequential(*[
            TfBlock(
                dim=self.embedding_dim,
                n_heads=self.head_num,
                dropout=self.dropout,
                attn_weight_dropout=0.0, # what should this number be?
                qkv_bias=False,
                dim_feedforward=self.ffn_embedding_dim
            ) for _ in range(self.encoder_layers)
        ]))
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
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

def train_sn_tokenizer():
     # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Customize training
    trainer = trainers.BpeTrainer(vocab_size=6, min_frequency=1, special_tokens=[
    ])
    tokenizer.train(files=["rankd_unsmoothed/chr1_unsmoothed_str.txt"], trainer=trainer)
    output = tokenizer.encode("ATCGN NCGG")
    print(output.tokens)
    tokenizer.save("sn_tokenizer.json")

# this is for a single chromosome
class TextDataset(Dataset):
    # sequence_length=1024, stride=512
    def __init__(self, filename, sequence_length=65536//64, stride=65536//128):
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
        
def collate_fn(batch):
    # Convert characters to integers
    batch = [sample for sample in batch]
    return batch

# could we do this such that the minibatches are well mixed?
def merge_train_set(filelist, merged_file_name):
    # List of files to merge: filelist
    
    # Open the output file in write mode
    with open(merged_file_name, "w") as outfile:
        # Iterate over the list of files
        for filename in filelist:
            # Open each file in read mode
            with open(filename) as infile:
                # Read the contents of the file
                contents = infile.read()
                # Write the contents to the output file, followed by a space
                outfile.write(contents + " ")

def epoch(model, rank, criterion,
               world_size, train_loader, use_wandb,
               optim, epoch_num, use_fsdp, tokenizer, accumulation_steps = 4):
    
    if use_fsdp:
        # need to divide by total gpu number to get local number of accumulation steps
        accumulation_steps = int(accumulation_steps / world_size)
        ddp_loss = torch.zeros(2).to(rank)

    accumulation_counter = 0
    total_loss = 0 

    print("number of batches in train loader", len(train_loader))
    raise Error
    for batch_i, (data) in enumerate(train_loader):
        with autocast(dtype=torch.bfloat16):
            #encoded_sequence = torch.tensor(tokenizer.encode(data[0]).ids, dtype=torch.long).to(rank) #[:seq_len
            # batch encoding
            #print(tokenizer.encode_batch(data))
            # Assuming `tokenizer` is an instance of `tokenizers.Tokenizer`
            encoded_sequence = tokenizer.encode_batch(data)
            
            # Convert the encoded sequences to tensors
            encoded_sequence = [torch.tensor(seq.ids, dtype=torch.long) for seq in encoded_sequence]
            
            # Stack the tensors into a single tensor
            encoded_sequence = torch.stack(encoded_sequence).to(rank)
            #print(encoded_sequence.shape)
            #encoded_sequence = encoded_sequence.unsqueeze(0)
            #print(encoded_sequence.shape)
            output = model.forward(encoded_sequence)
            logits, embeddings = output["logits"], output["embeddings"]
            logits = logits.permute(0, 2, 1)
            # so labels need to be torch.Size([1, N, vocab_size])
            #vocab_size = tokenizer.get_vocab_size()  # Get the size of the vocabulary
            #labels = F.one_hot(encoded_sequence, num_classes=vocab_size)
            #print("encoded_sequence", encoded_sequence.shape)
            #print("labels", labels.shape)
            # compute the loss
            # what is true here?
            loss = criterion(logits, encoded_sequence)
            #print(loss)
            print(batch_i, len(data[0]), encoded_sequence.shape)
        # normalize loss to account for batch accumulation
        #loss = loss / accumulation_steps
        loss.backward()
        print(loss)
        total_loss += loss.item()
        optim.step()
        optim.zero_grad(set_to_none=True)

        # catch gradients for wandb tracking
        # this probably works for single GPU too
        # model_names = []
        # grad_dict = dict()
        # for name, param in model.named_parameters():
        #     if param.requires_grad and param.grad is not None:#  param.grad is not None:
        #         # Gather the gradients from all GPUs
        #         grad = param.grad #.sum()
        #         #print(name, grad.shape)
        #         zero_threshold = 1e-6 
        #         grad_near_zero_count = torch.sum(grad.detach().cpu().abs() < zero_threshold).item()
        #         grad_total_count = grad.detach().cpu().numel()
        #         grad_near_zero_fraction = grad_near_zero_count / grad_total_count
        #         # grad_mean = torch.mean(grad.detach().cpu()).item()
        #         # grad_std = torch.std(grad.detach().cpu()).item() if grad_total_count > 1 else 0
        #         one_grad = {
        #             # f"gradients/{name}_mean": grad_mean,
        #             # f"gradients/{name}_std": grad_std,
        #             f"gradients/{name}_near_zero_fraction": grad_near_zero_fraction,
        #             }
        #         grad_dict.update(one_grad)
        #         #print(rank, type(grad_dict))
        #         model_names.append(name)
        # # log the gradients in wandb
        # if len(model_names) != 0 and use_wandb:
        #     wandb.log({f"gradients/{name}_{stat}": grad_dict[f"gradients/{name}_{stat}"] for name in model_names for stat in [ "near_zero_fraction"]}) 
        # if use_wandb:
        #     # track the activations via stats and histogram over time
        #     zero_threshold = 1e-6 
        #     matmask_near_zero_count = torch.sum(output["matmask"].abs() < zero_threshold).item()
        #     matmask_total_count = output["matmask"].numel()
        #     matmask_near_zero_fraction = matmask_near_zero_count / matmask_total_count

        #     wandb.log({
        #         "epoch": epoch_num,
        #         "train_loss": loss.item(),
        #         "epoch_progress": accumulation_counter / len(train_loader),
        #         "sample_seq_len": len(encoded_sequence),
        #         "matmask_activation_mean": torch.mean(output["matmask"].detach().to(torch.float32).cpu()).item(),
        #         "matmask_activation_std": torch.std(output["matmask"].detach().to(torch.float32).cpu()).item(),
        #         "matmask_activation_near_zero_fraction": matmask_near_zero_fraction,
        #     })

        if use_fsdp:
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data) 
    if use_fsdp:   
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    print(f"train loss: {total_loss/(batch_i+1)}")

# this one is for simple training
def single_GPU_main(train_path, val_path, epoch_num, model, optim, criterion, use_wandb, tokenizer):
    rank = 0 # cuda device zero
    world_size = 1
    model = model.to(rank)

    # check if bf16
    model = model.bfloat16() 

    dataset1 = TextDataset(train_path)
    sampler1 = torch.utils.data.RandomSampler(dataset1)
    train_loader = torch.utils.data.DataLoader(
                                                dataset1,
                                                collate_fn=collate_fn,
                                                batch_size=32,
                                                pin_memory=True,
                                                sampler=sampler1,
                                                drop_last=False
                                                )
    
    dataset2 = TextDataset(val_path)
    sampler2 = torch.utils.data.RandomSampler(dataset2)
    validation_loader = torch.utils.data.DataLoader(
                                                dataset2,
                                                collate_fn=collate_fn,
                                                batch_size=32,
                                                pin_memory=True,
                                                sampler=sampler2,
                                                drop_last=False
                                                )
    use_fsdp = False

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5 # patience
    best_epoch = 0

    for e in range(epoch_num):
        #for batch_i, (data) in enumerate(train_loader):
        epoch(model, rank, criterion,
               world_size, train_loader, use_wandb,
               optim, e, use_fsdp, tokenizer)
        

        # save a model checkpoint
        torch.save(model.state_dict(), f'model_checkpoint_{e}_rankd.pth')
        print(f"model saved at model_checkpoint_{e}_rankd.pth")

        # I want a validation loss check here. Given 70,000 samples, let's hold out 1000 randomly
        val_loss = validate(validation_loader, model, criterion, tokenizer, rank)
        # Check if the validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = e
        else:
            epochs_no_improve += 1
            # Check if the validation loss has not decreased for 5 epochs
            if epochs_no_improve == n_epochs_stop:
                print(f'Early stopping! Reached epoch {e+1} with best validation loss of {best_val_loss} at epoch {best_epoch}')
                break  # Stop training


def setup(rank, world_size, master_addr = 'localhost'): #"10.55.5.20"):
    os.environ['MASTER_ADDR'] = master_addr 
    os.environ['MASTER_PORT'] = '22'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ddp_main(rank, world_size, train_path, val_path,epoch_num, criterion, model, optim, tokenizer):
  
    model = model.bfloat16() 
    # set up the nccl process group
    setup(rank, world_size)                                   

    dataset1 = TextDataset(train_path)
    sampler = DistributedSampler(dataset1, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    
    train_loader = DataLoader(dataset1, collate_fn=collate_fn,
                            batch_size=32, pin_memory=True, num_workers=1, 
                            drop_last=False, shuffle=False, sampler=sampler)
    use_wandb, use_fsdp = False, False
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

        
    for i in range(epoch_num):
        if rank == 0:
            print(f"Epoch {i}")
        # if we are using DistributedSampler, we have to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch)  
        epoch(model, rank, criterion,
               world_size, train_loader,  use_wandb,
               optim, epoch_num, use_fsdp, tokenizer)
        cleanup()


        # # save a model checkpoint
        # torch.save(model.state_dict(), f'model_checkpoint_{e}.pth')
        # print(f"model saved at model_checkpoint_{e}.pth")

        # # I want a validation loss check here. Given 70,000 samples, let's hold out 1000 randomly
        # val_loss = validate(validation_loader, model, criterion, tokenizer, rank)
        # # Check if the validation loss has decreased
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        #     best_epoch = e
        # else:
        #     epochs_no_improve += 1
        #     # Check if the validation loss has not decreased for 5 epochs
        #     if epochs_no_improve == n_epochs_stop:
        #         print(f'Early stopping! Reached epoch {e+1} with best validation loss of {best_val_loss} at epoch {best_epoch}')
        #         break  # Stop training

def main():

    # Parse the arguments
    #args = get_args(ArgumentParser()).parse_args()
    criterion = nn.CrossEntropyLoss()
    
    epoch_num = 1
    use_wandb = False
    multi_GPU = False
    WORLD_SIZE = torch.cuda.device_count()


    # Create a Dataset
    all_other_chr =  [
        "phastcons_smoothed/chr3_smoothed_str.txt",          # "rankd_unsmoothed/chr3_unsmoothed_str.txt",
        "phastcons_smoothed/chr4_smoothed_str.txt",          # "rankd_unsmoothed/chr4_unsmoothed_str.txt",
        "phastcons_smoothed/chr5_smoothed_str.txt",          # "rankd_unsmoothed/chr5_unsmoothed_str.txt",
        "phastcons_smoothed/chr6_smoothed_str.txt",          # "rankd_unsmoothed/chr6_unsmoothed_str.txt",
        "phastcons_smoothed/chr7_smoothed_str.txt",          # "rankd_unsmoothed/chr7_unsmoothed_str.txt",
        "phastcons_smoothed/chr8_smoothed_str.txt",          # "rankd_unsmoothed/chr8_unsmoothed_str.txt",
    ]
    filelist = glob.glob("phastcons_smoothed/chr1*str.txt") + glob.glob("phastcons_smoothed/chr2*str.txt") + all_other_chr
   
    # merged_file_name = "phastcons_smoothed/all_but_chr9_str.txt"
    # merge_train_set(filelist, merged_file_name)
    train_path, val_path = "rankd_unsmoothed/all_but_chr9_str.txt", "rankd_unsmoothed/chr9_unsmoothed_str.txt"
    
    tokenizer = Tokenizer.from_file("sn_tokenizer.json") 
    print((tokenizer.get_vocab_size()))
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
    # for name, param in model.named_parameters():
    #     print(f"Name: {name}")
    #     print(f"Parameter: {param}")
    #     print(f"Parameter shape: {param.shape}")
    #     print("\n")

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
        mp.spawn(ddp_main,
                args = (WORLD_SIZE, train_path, val_path,
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
