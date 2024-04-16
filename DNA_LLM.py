# attention from scratch

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# from matplotlib import pyplot as plt
import time
from argparse import ArgumentParser
import math
from DNA_dist_utils import single_GPU_main, fsdp_main
from tokenizers import Tokenizer
import torch.multiprocessing as mp
import wandb

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
    def __init__(self, tokenizer, encoder_layers = 5, 
                 embedding_dim = 1024, ffn_embedding_dim = 5120,  
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
        encoder_block = nn.TransformerEncoderLayer(d_model = self.embedding_dim, 
                                                 nhead = self.head_num, 
                                                 dim_feedforward = self.ffn_embedding_dim,
                                                 activation ="gelu",
                                                 batch_first = True, # expects batch dim first
                                                 norm_first = True, # more stable
                                                 dropout = self.dropout)
        # don't we need the q and k heads exposed for rotary encoding?
        self.whole_encoder = nn.TransformerEncoder(encoder_block, 
                                             num_layers = self.encoder_layers)
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
    parser.add_argument("--chunk-dir", required = True, help = "dir where chr chunks are")# directory")
    parser.add_argument("--epoch", default = 2, type = int, help = "Number of epochs")
    # need more args
    return parser

# we should have everything we need now. Now we just need to debug to see that it can train


def main():
    # Parse the arguments
    args = get_args(ArgumentParser()).parse_args()
    criterion = nn.CrossEntropyLoss()
    
    epoch_num = 1
    use_wandb = False
    multi_GPU = False
    WORLD_SIZE = torch.cuda.device_count()

    tokenizer = Tokenizer.from_file("transcript_tokenizer.json")#"chr1_tokenizer.json")
    # you need to add all the special tokens to tokenizer
    special_tokens = ['<PAD>', '<MASK>', '<CLS>', '<EOS>']
    num_added_toks = tokenizer.add_tokens(special_tokens)

    #print((tokenizer.get_vocab_size())) # 30,002

    # you could still do flash attention with Rotary Encoding, you would just need to SDPA directly
    # that would expose the heads to do it 
    # need to init the model
    model = ConstraintBertModel(tokenizer)
    #print([param for param in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
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

        single_GPU_main(args.chunk_dir, epoch_num, model, optimizer, criterion, use_wandb, tokenizer)

        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()