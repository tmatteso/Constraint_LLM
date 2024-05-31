# compute LLR from the model

 # with torch.no_grad():
 #      logits = torch.log_softmax(model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from SN_BERT import ConstraintBertModel
from torch.cuda.amp import autocast
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
# load the model into memory from the checkpoint


def check_model_on_cuda(model):
    return all(param.is_cuda for param in model.parameters())

def split_clinvar(all_clinvar):
    # split into coding versus noncoding
    coding, noncoding = [], []
    
    for file in all_clinvar:
        #print(file)
        if file.split(".")[0].split("_")[-1] == "coding":
            coding.append(file)
        elif file.split(".")[0].split("_")[-1] == "noncoding":
            noncoding.append(file)   
        else:
            raise ValueError(f"{file} is neither coding nor noncoding")
    
    return coding, noncoding


def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences = [sample for sample in sequences]
    labels = torch.tensor(labels)
    return {"seqs": sequences, "labels": labels}

def chromo_extract(all_clinvar, model, tokenizer):
    label_arr, embedding_arr = [], []
    # Define batch size
    # batch_size = 32  # Adjust this value according to your GPU memory
    
    # # Prepare data
    # labels = []
    # sequences = []
    # for clinvar_str in all_clinvar:
    #     label = int(clinvar_str.split(".")[0].split("_")[-2])
    #     with open(clinvar_str, 'r') as file:
    #         clinvar = file.read()
    #         #encoded_sequence = tokenizer.encode(clinvar)
    #         if len(clinvar) > 1024:
    #             continue
    #         sequences.append(clinvar) #torch.tensor(clinvar, dtype=torch.long))
    #         labels.append(label)

    # # Create a DataLoader
    # data_loader = DataLoader(list(zip(sequences, labels)), batch_size=batch_size, collate_fn=collate_fn)
    
    # # Process batches
    # model = model.to('cuda')
    # all_labels, all_embeddings = [], []
    # for batch in data_loader:
    #     with autocast(dtype=torch.bfloat16):
    #         with torch.no_grad():
    #             # Assuming `tokenizer` is an instance of `tokenizers.Tokenizer`
    #             encoded_sequence = tokenizer.encode_batch(batch["seqs"])
                
    #             # Convert the encoded sequences to tensors
    #             encoded_sequence = [torch.tensor(seq.ids, dtype=torch.long) for seq in encoded_sequence]
                
    #             # Stack the tensors into a single tensor
    #             encoded_sequence = torch.stack(encoded_sequence).to(rank)
    #             #print(encoded_sequence.shape)
    #             #encoded_sequence = encoded_sequence.unsqueeze(0)
    #             #print(encoded_sequence.shape)
    #             output = model.forward(encoded_sequence)
    #             logits, embeddings = output["logits"], output["embeddings"]
    #             logits = logits.permute(0, 2, 1)
                
    #             print(embeddings.shape)
    #             raise Error
    #             inputs = {k: v.to('cuda') for k, v in batch.items() if k != 'labels'}
    #             output = model.forward(**inputs)
    #             logits, embeddings = output["logits"], output["embeddings"]
    #             mean_embedding = torch.mean(embeddings, dim=1).flatten()
    #             all_labels.extend(batch['labels'].cpu().numpy())
    #             all_embeddings.extend(mean_embedding.cpu().numpy())
    
    # label_arr = np.array(all_labels)
    # embedding_arr = np.array(all_embeddings)
    for i in range(len(all_clinvar)):
        clinvar_str = all_clinvar[i]
        label = int(clinvar_str.split(".")[0].split("_")[-2])
        with open(clinvar_str, 'r') as file:
            clinvar = file.read()
            with autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    encoded_sequence = tokenizer.encode(clinvar)
                    encoded_sequence = torch.tensor(encoded_sequence.ids, dtype=torch.long)
                    encoded_sequence = encoded_sequence.unsqueeze(0).to(0)
                    if (encoded_sequence).shape[1] > 1024:
                        # may want to slice the sequence and center it in the context later. For now just skip
                        continue
                   
                    model = model.to('cuda')
                    output = model.forward(encoded_sequence)
                    logits, embeddings = output["logits"], output["embeddings"]
                    mean_embedding = torch.mean(embeddings, dim=1).flatten()
                    label_arr.append(label)
                    embedding_arr.append(mean_embedding.cpu().numpy())
                    # label_arr[i] = label
                    # embedding_arr[i] = mean_embedding.cpu().numpy()
        print(i/len(all_clinvar))
    label_arr = np.array(label_arr)
    embedding_arr = np.array(embedding_arr)
    return label_arr, embedding_arr

# def process_chromo(chromo, all_clinvar, model, tokenizer):
#     clinvar_subset =[s for s in all_clinvar if chromo in s]
#     label_arr, embedding_arr = chromo_extract(all_clinvar, model, tokenizer)
#     return label_arr, embedding_arr

def extract_labels_and_embeds(all_clinvar, model, tokenizer):
    # init an array to store all labels and mean_embeddings
    label_arr = []#np.zeros(len(all_clinvar))
    embedding_arr = []#np.zeros((len(all_clinvar), 1024))
    
    # I would parallelize this by chromosome
    acceptable_contigs = [
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20, 21, 22
    ]
    all_labels, all_embeddings = [], []
    # clinvar_strs/phastcons_smoothed/chr9/77316382_1138826_0_coding.txt
    for i in range(len(acceptable_contigs)):
        chromo = "chr"+str(acceptable_contigs[i])
        clinvar_subset =[s for s in all_clinvar if chromo in s]
        label_arr, embedding_arr = chromo_extract(all_clinvar, model, tokenizer)
        all_labels.append(label_arr), all_embeddings.append(embedding_arr)
    
    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)
    return all_labels, all_embeddings

def eval_embeds(embedding_arr, label_arr):
    # Assuming `embeddings` is your array of embeddings and `labels` is your array of labels
    X_train, X_test, y_train, y_test = train_test_split(embedding_arr, label_arr, test_size=0.1, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    #y_pred = model.predict(X_test)

    # Assuming `model` is your trained model, `X_test` is your test set features, and `y_test` is your test set labels
    y_score = model.predict_proba(X_test)[:, 1]  # probabilities for the positive class
    
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # clean up and refactor stuff today.
    # try and get the coding vs. noncoding parts
    
    print(f'ROC-AUC: {roc_auc}')
    print(f'PR-AUC: {pr_auc}')
    print(f'positive class distribution: {len(y_test[y_test == 1]) / len(y_test)}')

def main():
    save = True
    # init the tokenizer
    tokenizer = Tokenizer.from_file("sn_tokenizer.json") 

    # you need to add all the special tokens to tokenizer
    special_tokens = ['<PAD>', '<MASK>', '<CLS>', '<EOS>'] # these are pointless. does your LLM really need this?
    num_added_toks = tokenizer.add_tokens(special_tokens)
    
    # Initialize the model
    model = ConstraintBertModel(tokenizer)
    
    # Load the checkpoint
    #checkpoint = torch.load("model_checkpoint_0_rankd.pth")#"../model_checkpoint_0_rankd.pth")
    checkpoint = torch.load('model_checkpoint_0_phastcons.pth')

    model.load_state_dict(checkpoint)
    print("model loaded")

    if save:
        # call the model on the clinvar strings
        #all_clinvar = glob.glob("clinvar_strs/rankd_unsmoothed/*/*.txt")
        all_clinvar = glob.glob("clinvar_strs/phastcons_smoothed/*/*.txt")
        
        # separate based on coding or noncoding
        coding, noncoding = split_clinvar(all_clinvar)
        print("coding split complete")
        # save coding
        coding_label_arr, coding_embedding_arr = extract_labels_and_embeds(coding, model, tokenizer)
        np.save('label_arr_coding.npy', coding_label_arr)
        np.save('embedding_arr_coding.npy', coding_embedding_arr)
        # save noncoding
        noncoding_label_arr, noncoding_embedding_arr = extract_labels_and_embeds(noncoding, model, tokenizer)
        np.save('label_arr_noncoding.npy', noncoding_label_arr)
        np.save('embedding_arr_noncoding.npy', noncoding_embedding_arr)
        print("arrays saved")

    else:
        coding_label_arr = np.load('label_arr_coding.npy')
        coding_embedding_arr = np.load('embedding_arr_coding.npy')
        noncoding_label_arr = np.load('label_arr_noncoding.npy')
        noncoding_embedding_arr = np.load('embedding_arr_noncoding.npy')
        print("arrays loaded")

    print("Coding:")
    eval_embeds(coding_embedding_arr, coding_label_arr)
    print("")
    print("Noncoding:")
    eval_embeds(noncoding_embedding_arr, noncoding_label_arr)
    print("")



    
    # {'<MASK>': 12, 
    #  '<pad>': 1, 
    #  'C': 6, 'A': 5, 'N': 8, 'Ġ': 10, 'T': 9
    #  '<EOS>': 14, 'G': 7, '<unk>': 3, '<mask>': 4, '<CLS>': 13, '<PAD>': 11, '<s>': 0, '</s>': 2}
    #tokens = [ '<s>', '<pad>', '</s>', '<unk>', '<mask>', 'A','C', 'G', 'N', 'T', 'Ġ', '<PAD>', '<MASK>', '<CLS>', '<EOS>']
    # with autocast(dtype=torch.bfloat16):
    #     with torch.no_grad():
    #         # Assuming `tokenizer` is an instance of `tokenizers.Tokenizer`
    #         encoded_sequence = tokenizer.encode(clinvar)
    #         print(encoded_sequence)
    #         # Convert the encoded sequences to tensors
    #         encoded_sequence = torch.tensor(encoded_sequence.ids, dtype=torch.long)
    #         encoded_sequence = encoded_sequence.unsqueeze(0).to(0)
    #         model = model.to('cuda')
    #         print(encoded_sequence.device)
    #         is_on_cuda = check_model_on_cuda(model)
    #         print(f"Model on CUDA: {is_on_cuda}")
    #         output = model.forward(encoded_sequence)
    #         logits, embeddings = output["logits"], output["embeddings"]
            # retrieve the embeddings and clinvar label to build a UMAP then train a logisitic regressor

    #         logits = logits.permute(0, 2, 1)
    #         # torch.Size([1, 15, 349])
    #         # B x classes x N
    #         llr = torch.log_softmax(logits, dim=1).cpu().numpy()
    #         # now get the ref allele for each position to find the WT LLR

    # print(tokens[5:10])
    # enc = encoded_sequence.cpu().numpy().flatten()
    # vals = []
    # for i in range(len(enc)):
    #     token_log = llr[:, :, i].flatten()
    #     WT_log = token_log[enc[i]]
    #     vals.append((token_log - WT_log)[5:10])

    # import numpy as np
    # vals = np.stack(vals)
    # print(vals.shape)
    # import matplotlib.pyplot as plt
    
    # # Assuming `tensor` is your 2D tensor
    # flattened_tensor = vals.ravel()
    
    # # Plot histogram
    # plt.hist(flattened_tensor, bins='auto')
    # plt.title('Histogram of Tensor Values')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.axvline(x=-13, color='r', linestyle='--')
    # plt.savefig("dist.png")
    # raise Error
    # print(llr.shape)
    # print(clinvar[:10])
    # print(encoded_sequence)
            #print(llr[:, :, 10:20].T)
            # CATGCTATCA
            # C -> 6
            # A -> 5
            # T -> 9
            # G -> 7
            # C -> 6
    # collect the logits (just output of constraint bert)

    # compute LLR



if __name__ == main():
    mp.set_start_method('spawn')
    main()