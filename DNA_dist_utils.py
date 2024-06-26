import torch
from torch import nn
import wandb
import os
import functools
from tqdm import tqdm
from torch.cuda.amp import autocast

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    ModuleWrapPolicy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, offload_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
    )

from DNA_data import DNA_dataset
import torch.nn.functional as F


def setup(rank, world_size, master_addr = 'localhost'): #"10.55.5.20"):
    os.environ['MASTER_ADDR'] = master_addr 
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, pdb_dir_path, epoch_num, criterion, model, optim, tokenizer):
    # set up the nccl process group
    setup(rank, world_size)  
    print(3)
    #wandb.init(project="constraint-bert")

    dataset1 = DNA_dataset(pdb_dir_path)
    sampler1 = DistributedSampler(dataset1, # this has no effect on making it load less than GPU number of samples
                                  rank=rank, 
                                  num_replicas=world_size, 
                                  shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset1,
                                                #collate_fn=collate_fn, # need to be defined
                                                batch_size=1,
                                                shuffle=False, 
                                                pin_memory=True,
                                                sampler=sampler1,
                                                drop_last=True)
    layers = {
              #nn.Embedding, # is this necessary?
              nn.TransformerEncoderLayer, # definitely necessary
              #LM_head # is this necessary?
              }
    # define the fsdp wrapping policy
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, recurse=True, transformer_layer_cls=layers#set([esm.model.esm2.ESM2, esm.esmfold.v1.trunk.FoldingTrunk])
    )
    torch.cuda.set_device(rank) # 
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    bfSixteen = MixedPrecision(
        # may help with layer norm? -- not sure, added autocast in the forward pass and that works by itself
        cast_forward_inputs = True, # actually each works separately
        param_dtype=torch.bfloat16, #torch.bfloat16,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16, #torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16, #torch.bfloat16,
        # keep gradients smaller
        #keep_low_precision_grads=True,
        )
    model = FSDP(model, 
                 use_orig_params = True, 
                 auto_wrap_policy = my_auto_wrap_policy, # this recursively wraps everything!
                 # a better wrapping policy may be the best option
                 # cpu offload actually makes you run out of memory faster. Is this a bug?
                 #cpu_offload=CPUOffload(offload_params=True), 
                 # CPU offload was messing up loss calculation -- still seems true -- now it doesn't
                 device_id=torch.cuda.current_device(),
                 mixed_precision=bfSixteen,
                 backward_prefetch = BackwardPrefetch.BACKWARD_PRE, # can try BACKWARD_POST to maybe save more mem
                 sync_module_states=True,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                )
    # now we apply an activation checkpoint wrap
    non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    # Lambda function to check if a submodule is an instance of any layer in the dictionary
    check_fn = lambda submodule: any(isinstance(submodule, layer_class) for layer_class in layers)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
    # now try torch compile -- doesn't play nice with fsdp!
    # model = torch.compile(model, mode="max-autotune")
    use_fsdp = True
    use_wandb = False
    # start the distributed processes
    init_start_event.record()
        
    for i in range(epoch_num):
        if rank == 0:
            print(f"Epoch {i}")
        epoch(model, rank, criterion,
               world_size, train_loader,  use_wandb,
               optim, epoch_num, use_fsdp, tokenizer)

    # end the distributed processes
    init_end_event.record()


# I think you can just write one function with an arg that controls FSDP or not
def epoch(model, rank, criterion,
               world_size, train_loader, use_wandb,
               optim, epoch_num, use_fsdp, tokenizer, accumulation_steps = 4):
    
    if use_fsdp:
        # need to divide by total gpu number to get local number of accumulation steps
        accumulation_steps = int(accumulation_steps / world_size)
        ddp_loss = torch.zeros(2).to(rank)

    accumulation_counter = 0
    total_loss = 0 

    for batch_i, (data) in enumerate(train_loader):
        #print("data", data) # should be str
        # collect data

        # increasing layer number to 10 does not affect memory!!!! for activation checkpointing and fsdp
        # more layers, loss go down faster

        # more layers, does increase memory usage for the non ac, non fsdp case

        # fix layer number to 5, 64 attention heads, embedding_dim = 8192, ffn_embedding_dim = 24576
        # fsdp + activation checkpointing + mixed precision: model dim = 8192, 65536 tokens is max
        # fsdp + mixed precision: model dim = 8192, 32768 tokens is max
        # full precision: model dim = 8192, 8192 tokens is max
        # having the model dim so high makes the loss go down muchhhhhhhhhh faster


        # fix layer number to 5, 64 attention heads, embedding_dim = 4096, ffn_embedding_dim = 12288
        # full precision: model dim = 4096, 32768 tokens is max
        # fsdp + mixed precision: model dim = 4096, 32768*2 tokens is max
        # fsdp + activation checkpointing + mixed precision: model dim = 4096, 32768 *4

        # fix layer number to 5, 32 attention heads, embedding_dim = 4096, ffn_embedding_dim = 12288
        # same as with 64 heads

        # fix layer number to 5, 16 attention heads, embedding_dim = 2048, ffn_embedding_dim = 12288 /2
        # the model is way worse now, loss goes down way slower
        # fsdp + activation checkpointing + mixed precision: 32768 *8 tokens is max
        # fsdp + mixed precision: 32768*2 tokens is max
        # full precision: 32768 tokens is max

        # once done with this, you need to check if all this even using flash attention



        # chinchilla estimate for 150B tokens is ~ 10B params, this is ~10% of Whole genome across 500 genomes
        # 7.5 B tokens is ~ 500M params, ~1% of whole genome across 250 genomes
        #seq_len = int(8192 *8)

        #print("encoded_sequence", encoded_sequence.shape) # should be torch.Size([1, N])
        # it wants logits to be torch.Size([1, vocab_size]) for CE loss
        # feed it through the model forward
        if len(data[0]) > 90000:
            print(batch_i, len(data[0]))
            continue

        with autocast(dtype=torch.bfloat16):
            encoded_sequence = torch.tensor(tokenizer.encode(data[0]).ids, dtype=torch.long).to(rank) #[:seq_len]
            encoded_sequence = encoded_sequence.unsqueeze(0)
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
        total_loss += loss.item()
        optim.step()
        optim.zero_grad(set_to_none=True)

        # if batch_i == 20:
        #     print(f"train loss: {total_loss/(batch_i+1)}")
        #     break



        # catch gradients for wandb tracking
        # this probably works for single GPU too
        model_names = []
        grad_dict = dict()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:#  param.grad is not None:
                # Gather the gradients from all GPUs
                grad = param.grad #.sum()
                #print(name, grad.shape)
                zero_threshold = 1e-6 
                grad_near_zero_count = torch.sum(grad.detach().cpu().abs() < zero_threshold).item()
                grad_total_count = grad.detach().cpu().numel()
                grad_near_zero_fraction = grad_near_zero_count / grad_total_count
                # grad_mean = torch.mean(grad.detach().cpu()).item()
                # grad_std = torch.std(grad.detach().cpu()).item() if grad_total_count > 1 else 0
                one_grad = {
                    # f"gradients/{name}_mean": grad_mean,
                    # f"gradients/{name}_std": grad_std,
                    f"gradients/{name}_near_zero_fraction": grad_near_zero_fraction,
                    }
                grad_dict.update(one_grad)
                #print(rank, type(grad_dict))
                model_names.append(name)
        # log the gradients in wandb
        if len(model_names) != 0 and use_wandb:
            wandb.log({f"gradients/{name}_{stat}": grad_dict[f"gradients/{name}_{stat}"] for name in model_names for stat in [ "near_zero_fraction"]}) #"mean", "std",]})

        # accumulation_counter += 1
        # if accumulation_counter % accumulation_steps == 0 or (batch_i + 1 == len(train_loader)):
        #     # Clip gradients
        #     #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        #     #print("step taken")
        #     optim.step()
        #     optim.zero_grad(set_to_none=True)
        #     if rank == 0:
        #         print(f"epoch: {epoch_num}, loss: {loss}, global_batch_size: {accumulation_steps*world_size}",
        #                flush=True)
                
        if use_wandb:
            # track the activations via stats and histogram over time
            zero_threshold = 1e-6 
            matmask_near_zero_count = torch.sum(output["matmask"].abs() < zero_threshold).item()
            matmask_total_count = output["matmask"].numel()
            matmask_near_zero_fraction = matmask_near_zero_count / matmask_total_count

            wandb.log({
                "epoch": epoch_num,
                "train_loss": loss.item(),
                "epoch_progress": accumulation_counter / len(train_loader),
                "sample_seq_len": len(encoded_sequence),
                "matmask_activation_mean": torch.mean(output["matmask"].detach().to(torch.float32).cpu()).item(),
                "matmask_activation_std": torch.std(output["matmask"].detach().to(torch.float32).cpu()).item(),
                "matmask_activation_near_zero_fraction": matmask_near_zero_fraction,
            })

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

    dataset1 = DNA_dataset(train_path)
    sampler1 = torch.utils.data.RandomSampler(dataset1)
    train_loader = torch.utils.data.DataLoader(
                                                dataset1,
                                                #collate_fn=collate_fn,
                                                batch_size=1,
                                                shuffle=False, 
                                                pin_memory=True,
                                                sampler=sampler1,
                                                #drop_last=True
                                                )
    
    dataset2 = DNA_dataset(val_path)
    sampler2 = torch.utils.data.RandomSampler(dataset2)
    validation_loader = torch.utils.data.DataLoader(
                                                dataset2,
                                                #collate_fn=collate_fn,
                                                batch_size=1,
                                                shuffle=False, 
                                                pin_memory=True,
                                                sampler=sampler2,
                                                #drop_last=True
                                                )
    use_fsdp = False

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = 5
    best_epoch = 0

    for e in range(epoch_num):
        #for batch_i, (data) in enumerate(train_loader):
        epoch(model, rank, criterion,
               world_size, train_loader, use_wandb,
               optim, e, use_fsdp, tokenizer)
        

        # save a model checkpoint
        torch.save(model.state_dict(), f'model_checkpoint_{e}.pth')
        print(f"model saved at model_checkpoint_{e}.pth")

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

                
# right now we have ~ 562,949,953 params! This is pretty much chinchilla optimal

def validate(validation_loader, model, criterion, tokenizer, rank):
    # Calculate the validation loss
    val_loss = 0
    model.eval()  # Set the model to evaluation mode


    with torch.no_grad():  # Disable gradient calculations
        for batch_i, (data) in (enumerate(validation_loader)):
            #print("encoded_sequence", encoded_sequence.shape) # should be torch.Size([1, N])
            # it wants logits to be torch.Size([1, vocab_size]) for CE loss
            # feed it through the model forward
            with autocast(dtype=torch.bfloat16):
                encoded_sequence = torch.tensor(tokenizer.encode(data[0]).ids, dtype=torch.long).to(rank)#[:seq_len]
                encoded_sequence = encoded_sequence.unsqueeze(0)
                output = model.forward(encoded_sequence)
                logits, embeddings = output["logits"], output["embeddings"]
                logits = logits.permute(0, 2, 1)
                # so labels need to be torch.Size([1, N, vocab_size])
                loss = criterion(logits, encoded_sequence)
            #print(loss)
            val_loss += loss.item()  # Accumulate the loss

    print(f'Validation loss: {val_loss/(batch_i+1)}')
    return val_loss
            # if batch_i == 20:
            #     print(f'Validation loss: {val_loss/(batch_i+1)}')
            #     return val_loss

