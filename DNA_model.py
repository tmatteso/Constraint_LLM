# you need to write this model from scratch first

# slow down. take it step by step

# tokenize str (seq_len x vocab_size)

# embed the tokens (seq_len x embed_size)


################################################################
################## Scaled Dot Product (Self) Attention #########
################################################################

# apply linear layer (Q: embed_size, hidden_size) to embedddings
# Embeddings x Q -> Linear Projection of Q (seq_len, hidden_size)

# do the same with another linear layer (K: embed_size, hidden_size)
# Embeddings x K -> Linear Projection of K (seq_len, hidden_size)

# transpose K, (Proj_of_K.T: (hidden_size,seq_len))

# multiply Q (seq_len, hidden_size) and K.T (hidden_size,seq_len) -> M (seq_len, seq_len)

# normalize by the square root of the hidden_size (M/ sqrt(hidden_size))

# apply softmax to get attention scores: softmax((M/ sqrt(hidden_size)) 
# what exactly is a softmax?

# apply linear layer (V: embed_size, hidden_size) to embedddings
# Embeddings x V -> Linear Projection of V (seq_len, hidden_size)

# multiply Attention scores (seq_len, seq_len) and Proj_of_V (seq_len, hidden_size) -> new_embeds (seq_len, hidden_size)

# and now you have the new embeds with context info (seq_len, hidden_size);


################################################################
################## Multi Head Attention ########################
################################################################

# concatenate the output of N many heads (head_number) of Independent Self Attention Blocks
# concat_new_embeds (seq_len, hidden_size, head_number)
# generally Self Attention Block has hidden_size = embed_size / head_number

# then a linear layer (Output) is fired on the concat_new_embeds
# Output (embed_size, embed_size) x concat_new_embeds (seq_len, embed_size, head_number)

# and we get MHA contextualized new_embeds (seq_len, hidden_size)

################################################################
################## Feed Forward Network ########################
################################################################

# two layer NN applied to each token embedding in the batch and sequence independently
# this ensures that the contextual embeddings are distinct and spread out
# generally is LL -> LL -> GeLU activation -> Dropout


################################################################
################## Layer Normalization #########################
################################################################

# this operations stabilizes training; prevents exploding and vanishing gradients
# it normalizes the activations for each sequence, preventing them from getting too small or too large

# compare against Batch Normalization and explicitly spell out the operation

# Batch normalization normalizes each feature independently across the mini-batch.
# Layer normalization normalizes each of the inputs in the batch independently across
# all features. Batch norm works across batch dimension of activations, layer norm works 
# across feature dimension of each activation. Layer norm is independent of Batch size so 
# it is better for sequence models at inference because batch sizes can be small and sequence 
# lengths can vary. Layer norm does not need to be adjusted at inference, batch norm has to 
# have batch statistics stored to work properly. 


# LN originally came at the end of a Transformer Block but now comes in the beginning
# Old Transformer Encoders: SAs -> add(MHA(SAs), residuals of SAs): MHA+ -> LN: LN_MHA+ -> add(FFN(LN_MHA+), residuals of LN_MHA+) -> LN -> new_embeds (seq_len, embed_size)
# and we have contextualized new_embeds (seq_len, embed_size)

# TRUE END OF TRANSFORMER BLOCK!


# New Transformer Encoders:
# LN(SAs): LN_SAs -> MHA(LN_SAs): MHA -> add(MHA, residuals of SAs): MHA+ -> LN: LN_MHA+ -> add(FFN(LN_MHA+), residuals of LN_MHA+) -> LN -> new_embeds (seq_len, embed_size)
# and we have contextualized new_embeds (seq_len, embed_size)

# TRUE END OF TRANSFORMER BLOCK!


################################################################
################## Positional Encoding #########################
################################################################

# Sinuisodal Positional Encoding (Absolute Positions)
# static sinuisoidal function, does not generalize beyond context length it was trained on
# They are a function of given position i, a given dimension j in the sequence, and the dimension of the model

# Alibi

# Rotary Positional Encodings (Absolute + Relative Positions)
# literally involves rotation matrices on the embedding vectors: keeps magnitude but changes angles
# we use multiple blocks of rotation matrices to do RPE
# use different 2-D rotation matrices for each pair of dimensions (every set of vectors of size 2) by some angle
# this means our embedding dimension must be divisible by 2
# embed_dim / 2 = number of rotation matrices
# theta = 10,000 ** (2(i-1)/d) 
# where i = 1,2, ... d/2 (for each pair of features) and d is equal to embedding dimension
# then we simply apply these theta for each pair of dimensions
# m = index of token in the sequence 
# m_vector * theta_vector = angle_vector (rotation matrix)
# apply the angle vector on the pairwise dimension vectors p
# p * angle vector = rotated pairwise embeddings

# this means that each word is rotated by embedding size / 2 pairs of rotation matrices
# each with different rotations that are affected by the positions of each word

# so each positional encoding will take the word’s position into account, 
# and we will now be able to generate different attention outputs for the same word in different positions

# what's really great is that the dot product between pairs of token embeddings stay the same regardless of position, as long as they are at the same position
# also if the positional encoding of a word has a larger angle of rotation, the word is further along the sentence

# how is this learned?

# YaRN
# YaRN reaches state-of-the-art performance in context window extensions after fine-tuning on less than ∼0.1% of the original pre-training data.
# we use a temperature param before the softmax for self attention
# sqrt(1 / t) = 0.1*ln(s) + 1
# where s = L' / L where  L’ is the fixed number of extended context size, L is the old context size
# softmax(Q.T x K / sqrt(|D|)) - > softmax(Q.T x K / (t *sqrt(D)))
# this scales attention  by the amount we are increasing the context window length
# directly compatible with flash Attention 2!

# you just need to backprop to let attention readjust to the scale factor


################################################################
########## Encoder Only Full Transformer Architecture ##########
################################################################

# input tokenized sequence

# embed the tokens and add positional encodings -> embeds (seq_len, embed_size)

# However many Transformer Encoder Blocks: TEB(TEB(TEB(embeds))) -> new_embeds (seq_len, embed_size)

# get embedding at first position: new_embeds[0] (1, embed_size)

# maybe apply dropout (depends on architecture)

# apply a Linear Layer (embed_size, vocab_size) to get Logits

# apply Softmax to Logits (1, vocab_size) -> probabilities for each token in vocab_size (class)



################################################################
########## Decoder Only Full Transformer Architecture ##########
################################################################

# input tokenized sequence

# embed the tokens and add positional encodings -> embeds (seq_len, embed_size)

# then apply LN on embeds (seq_len, embed_size) -> embeds_LN (seq_len, embed_size)

# apply MHA on embeds_LN (seq_len, embed_size) -> MHA_embeds_LN (seq_len, embed_size)

# add(MHA_embeds_LN, pass through of embeds) -> new_embeds (seq_len, embed_size)

# then apply LN on new_embeds (seq_len, embed_size) -> new_embeds_LN (seq_len, embed_size)

# apply FFN to new_embeds_LN -> FFN_new_embeds_LN (seq_len, embed_size)

# add(FFN_new_embeds_LN, pass through of new_embeds) -> final_embeds (seq_len, embed_size)

# TRUE END OF TRANSFORMER BLOCK!

# but masks? apply lower triangular binary matrix mask to MHA_embeds_LN
# replace non masked elements with negative infinity
# after softmax these attention scores will be zero!

# WITH CAUSAL MASK FOR SELF ATTENTION:

# comes after (Q x K.T) / sqrt(hidden_size)

# apply Lower Triangular Matrix

# apply Softmax to get attention scores (seq_len, seq_len)

# SAMPLING SEQUENCES FROM A DECODER:

# get first position of final_embeds (seq_len, embed_size) -> final_embeds[0] (1, embed_size)

# apply a Linear Layer (embed_size, vocab_size) -> logits (1, vocab_size)

# apply temperature scaling (softmax param) or top K filtering to subset the Logit vector

# take the softmax of the transformed logits -> Next token probabilities 

# filter by Probabilities (Top P) if you desire

# sample from probabilities (sample from multinomial, seems like multinoulli) -> next token

# run sequence embedding and positional encoding, put through the Decoder blocks again

# REPEAT!

################################################################
################# Grouped Query Attention ######################
################################################################

# faster batched inference, share weights across heads when doing proj of K and V
# nheads, (V: embed_size, hidden_size) -> (n, k) (n, v)???

# GQA interpolates between multi query attention and multi head attention
# it tries to achieve the quality if MHA while maintaining the speed of MQA
# GQA divides query heads into G groups, each of which shares a single key head and value head

# you can group queries based on similarity
# there's a lot you can do here. I think you should avoid this in implementation




################################################################
########################## BERT LOSS ###########################
################################################################

# 15% of tokens are replaced with [MASK] token
# then model attempts to predict token was there before (classification task with vocab_size many classes)
# so we add a classification layer on top of the encoder output (this is the final LL)
# we then multiply output vects by embedding matrix, transforming them into the vocab_size
# calculate the prob of each word in the vocab with softmax

# so we use the CrossEntropy loss, as this is a classification task
# cross entropy is done with probability distributions!
# as predicted probability diverges from actual label, CE loss increases
# perfect moel woud have a log loss of 0
# Prob= 0, log(0) -> inf
# Prob = 1, log(1) -> 0
# so small logs are big probs and vice versa. 
# in BERT we condition upon all unmasked tokens
# so sum (over unmasked seq len) log P(masked token | unmasked tokens)
# specifically penalizes those predictions that are confident and wrong


################################################################
########################## GPT LOSS ############################
################################################################

# sum (over seq len) log P(next_token | all previous tokens)
# still cross entropy, just a different masking on the sequence before classification


################################################################
############### ALPHA MISSENSE LOGIT LOSS ######################
################################################################

# after pretraining, we attempt to align logit scoring with pathogenicity labels

# s_a_i is the logit score of the alternative allele compared to reference at position i
# s_a_i = log (p_ref_i) - log (p_a_i)

# during training, masked variants are sampled and included in the variant embedding
# each variant gets a label y_i {0 if benign, 1 if sampled from unobserved set} and a weight w_i
# the training loss function for pathogenicity logit scores is:
# Loss = (1/N) sum_for_each_i( w_i*( (y_i)*log(sigmoid(s_i | > -1)) + 
#                                    (1-y_i)*log(sigmoid(s_i | > 0)) ) )
# where N is the number of positions (i, i+1, ..., N)
# s_i | > -1 is max(s_i, -1)
# s_i | > 0 is max(s_i, 0)

# we could use cancer variants from the 100,000 Cancer Genomes Project


################################################################
############# CALIBRATION / CONFIDENCE LOGIT RESCALING #########
################################################################

# train a linear logistic rescaling function: literally split the space with a line
# if far from line in negative direction: benign
# if far from line in positive direction: pathogenic

# c_1 * s + c_0 
# where c_1 and c_0 are learnable scalars to modify the logit score s

# then rescale the distance with a sigmoid to compress to [0, 1] Domain
# sigmoid(c_1 * s + c_0)



################################################################
############### FLASH ATTENTION V1 and V2 ######################
################################################################

# Fast — excerpt from the paper: “We train BERT-large (seq. length 512) 15% faster 
# than the training speed record in MLPerf 1.1, GPT2 (seq. length 1K) 
# 3x faster than baseline implementations from HuggingFace and Megatron-LM, 
# and long-range arena (seq. length 1K-4K) 2.4x faster than baselines.”
# Memory-efficient — compared to vanilla attention, which is quadratic in sequence length, O(N²),
# this method is sub-quadratic/linear in N (O(N)). We’ll see later why & how.
# Exact — meaning it’s not an approximation of the attention mechanism 
# (like e.g. sparse, or low-rank matrix approximation methods) — 
# its outputs are the same as in the “vanilla” attention mechanism

# memory efficient:
# it minimizes I/O with HBM by removing redundant reads and writes
# keep everything in SRAM, compute all intermediate steps, and only after write to HBM
# this is accomplished by kernel fusion. putting all the steps together in one GPU operation
# this way you only write back to HBM once that operation is done

# materialization: in normal attention we've allocated full NxN matrices for the activations in memory
# with materialization we only allocated chunks of the NxN matrices as they are computed, greatly reducing allocation
# technically this makes attention O(N), not O(N^2) with respect to memory.

# at its core, flash attention is all about tiling and recomputation:

# Tiling is used during both backward and forward passes, this chunks the NxN the softmax matrix into blocks
# however we still have to compute the softmax across all tokens. How does tiling handle this?
# so we chop the softmax computation into smaller blocks and then compute the exact result 
# from the output of all these smaller blocks (need to keep track of intermediate stats tho)

# recomputation is used only the backward pass, very similar to activation checkpointing
# this is activation checkpointing:
# we have to have activations computed in the forward pass available for backward to compute grads
# so we don't store them during the forward pass, only what is needed for the next layer
# this comes at a cost of recomputing them during backward pass (slower cuz more matmuls)

# however in flash attention, recomputation comes with no cost to memory or runtime!
# by storing the output (N x d), the softmax normalization stats (N),
# we can recompute attention projection matrices S (N x N) and P (N x N) in the backward pass
# directly from blocks of Q, K, and V (N x d) in SRAM, keeping memory at O(N)!!

# however because it is so dependent on precise GPU pipelining, flash attention is not supported
# on all backends, or even all CUDA GPUs (like V100)

# Flash Attention2: with normal flash attention, training with 8k is as easy as training with 2k tokens!

# this number is without activation checkpointing or ZERO! (may include mixed precision)
# 32k is probably doable through normal means with A100s

# FA2 is a redo that reduces the number of non-matmul FLOPs, achieves ~2x speedup
# now it can also parallize over sequence length, not just batch dim
# can use multiple thread blocks for a sequence
# it also improves thread block partitioning to reduce the synchronization and comm between groups of 32 threads

# Fa1 only supported head dimensions up to 128, now FA2 goes up to 256
# FA2 also supports multiquery attention (MQA) and Grouped buery attention (GQA)
# these are attention variants where multiple heads of a query attend to the same 
# head of key and value, in order to reduce the size of KV cache during inference
# this increases inference throughput -- important for actual use lol

# FA1 and FA2 are implemented in pytorch 2.2 natively, in 
# TORCH.NN.FUNCTIONAL.SCALED_DOT_PRODUCT_ATTENTION, which is used in 
# torch.nn.TransformerEncoderLayer and torch.nn.MultiheadAttention
# you may need to check the context manager to make sure the right implementation is being used:
# torch.backends.cuda.flash_sdp_enabled() should return True
# torch.backends.cuda.sdp_kernel(enable_flash=True) can activate flash attention temporarily
# This context manager can be used to temporarily enable or disable any of the three backends 
# for scaled dot product attention. Upon exiting the context manager, the previous state of 
# the flags will be restored.

# project trajectory: 

# 1. basic BERT model with 1% subset of phyloP on primate Alignment + add confidence loss
# will use Byte Pair Encoding, Flash Attention, Phylogenetic Tokenization, Rotary Positional Encodings, Logit Loss finetuning, Calibration
# this is the pilot paper

# 2. extended BERT model with 5 or 10% subset of phyloP on mammalian Alignment (way more tokens)
# same as 1 but show scaling laws
# we need more compute for this one

# 3. YaRN finetuning the mammalian extended BERT
# now show the effect of enhancer masking for personalized and self type specific expression
# still need more compute


# Goals for Project 1:
# do PhyloP data science to get genome subset, chunk into smaller fastas
# need dataloader to read in each fasta chunk at a time
# Byte Pair Encoding -- done
# use chinchilla to get optimal param number and depth for token amount, also number of epochs
# do tuning to find optimal batch size and context length to max out memory
# do small search on adam params
# yeet it into a learning rate scheduler

# Validation for Project 1:
# keep in mind we only have Promoters+UTRs+CDS for each gene: best case CTCF bound regions, but not enhancers
# do coding Clinvar
# do non coding Clinvar
# splice variant scoring
# histone occupancy
# atacQTLs
# eQTLs
# Polygenic Risk Scores
# Prior for GWAS finemapping
# Recombination Rate
# COSMIC frequency vs gnomad common
# OMIM pathogenic vs. gnomad common
# odds ratio of gnomad rare vs. common
# correlation with gnomad low frequency variants
# recover secondary structure from exon subset of attention map?
# recover genome secondary structure from the whole attention map
# atac QTLs are more heritable and line up better with complex human disease





