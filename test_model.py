from ProtFlash_Man.pretrain import load_prot_flash_small
from ProtFlash_Man.utils import batchConverter
from PredictionHead.model import Convolution_Predictor
import torch
import torch.nn as nn
import pandas as pd

ss_dict = {'H':0, 'B':1, 'E':2, 'G':3, 'I':4, 'T':5, 'S':6, 'C':7}
examples = 128

# can modify this as convenient
device = torch.device("cuda")

# df = pd.read_csv('/Users/aidan/Documents/COS 398/archive/PDB_31-07-2011.csv', skiprows=1200, nrows=100)
df = pd.read_csv('/scratch/network/ap9884/PDB_31-07-2011.csv', nrows=1456)

plm_model = load_prot_flash_small().to(device)

model = Convolution_Predictor(512).to(device)
checkpoint = torch.load('/home/ap9884/cos-iw-protflash/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()

# test on a few in distribution
batch = df.tail(examples)

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
output = model(token_embedding, masks.to(device))

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
    print(loss.item())
else:
    print('did not match')

# test on a few out of distribution