# compute LLR from the model

 # with torch.no_grad():
 #      logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from SN_BERT import ConstraintBertModel
# load the model into memory from the checkpoint




def main():
    # init the tokenizer
    tokenizer = Tokenizer.from_file("../sn_tokenizer.json") 

    # you need to add all the special tokens to tokenizer
    special_tokens = ['<PAD>', '<MASK>', '<CLS>', '<EOS>']
    num_added_toks = tokenizer.add_tokens(special_tokens)
    
    # Initialize the model
    model = ConstraintBertModel(tokenizer)
    
    # Load the checkpoint
    checkpoint = torch.load("../model_checkpoint_0_rankd_only.pth")
        #'../model_checkpoint_0_phastcons_smoothed_only.pth')

    model.load_state_dict(checkpoint)
    print(model)
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Parameter: {param}")
        print(f"Parameter shape: {param.shape}")
        print("\n")

    # did the custom module weights not get saved?????
    pass


if __name__ == main():
    main()