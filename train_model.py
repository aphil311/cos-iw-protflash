from ProtFlash_Man.pretrain import load_prot_flash_small
from ProtFlash_Man.utils import batchConverter
from PredictionHead.model import Convolution_Predictor
import torch
import torch.nn as nn
import pandas as pd

# import tensorflow as tf
# import tensorflow_datasets as tfds

# from ProtFlash.model import FLASHTransformer

ss_dict = {'H':0, 'B':1, 'E':2, 'G':3, 'I':4, 'T':5, 'S':6, 'C':7}

# MODEL_URL_SMALL = "/scratch/network/ap9884/flash_protein.pt"
# MODEL_URL_SMALL = "./pretrained-models/flash_protein.pt"
examples = 256
training_steps = 5000
# examples = 1
# training_steps = 1

# set device to be gpu
device = torch.device("cuda")

# print('building dataframe...')
# read in proteins
# df = pd.read_csv('/Users/aidan/Documents/COS 398/archive/PDB_31-07-2011.csv', nrows=8)
df = pd.read_csv('/scratch/network/ap9884/PDB_31-07-2011.csv', nrows=1500)

plm_model = load_prot_flash_small().to(device)
model = Convolution_Predictor(512).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

checkpoint = torch.load('/scratch/network/ap9884/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# training loop
print('starting training loop...')
total_loss = 0
loss_count = 0
for s in range(training_steps):
    # print what iteration we are
    batch = df.sample(examples)

    data = []       # training data
    labels = []     # training labels
    lengths = []    # tracks length of each sequence
    max_len = 0     # longest sequence

    # populate the labels and find the max length (for batching and masking)
    for index, row in batch.iterrows():
        pdf_id = row['pdb_id']
        sst8 = row['sst8']
        lengths.append(len(sst8))
        labels.append((pdf_id, sst8))
    max_len = max(lengths)

    # populate the data and generate masks
    masks = torch.full((examples, 256, max_len), True, dtype=torch.bool).to(device)
    i = 0
    for index, row in batch.iterrows():
        pdf_id = row['pdb_id']
        seq = row['seq']
        masks[i, :, len(seq):] = False
        data.append((pdf_id, seq))
        i += 1

    # print('getting embeddings...')
    ids, batch_token, lengths = batchConverter(data)
    with torch.no_grad():
        token_embedding = plm_model(batch_token.to(device), lengths.to(device))

    # go from (N, L, d) to (N, d, L)
    token_embedding = torch.transpose(token_embedding, 1, 2)

    # Build the mask vector
    # print('getting secondary structures...')
    output = model(token_embedding, masks.to(device))
    # trimmed_output = []
    # pretty_output = []
    # i=0
    # for o in output:
    #     trimmed_output.append(o[:lengths[i], :])
    #     pretty_output.append(torch.argmax(o[:lengths[i], :], dim=1))
    #     i += 1

    # get a matrix (N, L, 8) that is one-hot
    targets = torch.full((examples, 8, max_len), 0, dtype=float).to(device)
    i=0
    for protein in labels:
        j=0
        for aa in protein[1]:
            index = ss_dict.get(aa)
            if index is None:
                print('Error:', aa, 'not found')
            targets[i, index, j] = 1
            j+=1
        i+=1

    if output.shape[2] == targets.shape[2]:
        loss = criterion(output, targets)
        total_loss += loss.item()
        loss_count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((s+1) % 100 == 0):
            print('epoch: ', s+1,' average loss: ', total_loss / loss_count)
            
            total_loss = 0
            loss_count = 0
    else:
        print('did not match')
    
    # print('epoch: ', s+1,' loss: ', loss.item())
    
    # # output_labels = torch.argmax(output, dim=2)
    # idx_to_label = {v:k for k, v in ss_dict.items()}

    # i=0
    # for l in pretty_output:
    #     # print('\n---------------------')
    #     final = [idx_to_label[i] for i in l.cpu().numpy()]
    #     string = ""
    #     i += 1
    #     # print(string.join(final))
    #     # print(labels[i][1])
    # if s == 10 or s == 9:
    #     print('epoch: ', s+1,' loss: ', loss.item())

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, '/scratch/network/ap9884/model2.pt')

df = pd.read_csv('/scratch/network/ap9884/PDB_31-07-2011.csv', nrows=1)

data = []
for index, row in df.iterrows():
    pdf_id = row['pdb_id']
    seq = row['seq']
    l = len(seq)
    data.append((pdf_id, seq))

masks = torch.full((1, 256, l), True, dtype=torch.bool).to(device)

print('getting embeddings...')
ids, batch_token, lengths = batchConverter(data)
with torch.no_grad():
    token_embedding = plm_model(batch_token.to(device), lengths.to(device))

# go from (N, L, d) to (N, d, L)
token_embedding = torch.transpose(token_embedding, 1, 2)

print('getting secondary structures...')
output = torch.transpose(model(token_embedding, masks.to(device)), 1, 2)
output_labels = torch.argmax(output, dim=2)
idx_to_label = {v:k for k, v in ss_dict.items()}
# print(output_labels[0])
# final_label_0 = [idx_to_label[i] for i in output_labels[0].cpu().numpy()]
# string = ""
# print(string.join(final_label_0))
for l in output_labels:
    final = [idx_to_label[i] for i in l.cpu().numpy()]
    string = ""
    print(string.join(final))

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, '/scratch/network/ap9884/model.pt')
