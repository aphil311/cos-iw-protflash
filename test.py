from ProtFlash_Man.pretrain import load_prot_flash_small
from ProtFlash_Man.utils import batchConverter
from PredictionHead.model import Convolution_Predictor
import torch
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

from ProtFlash.model import FLASHTransformer

# MODEL_URL_SMALL = "/scratch/network/ap9884/flash_protein.pt"
# MODEL_URL_SMALL = "./pretrained-models/flash_protein.pt"

print('building dataframe...')
examples = 16
df = pd.read_csv('/Users/aidan/Documents/COS 398/archive/PDB_31-07-2011.csv', nrows=examples)

data = []
lengths = []
# max_len = 1058
max_len = 536
masks = torch.full((examples, 256, max_len), True, dtype=torch.bool)
i = 0
for index, row in df.iterrows():
    pdf_id = row['pdb_id']
    seq = row['seq']
    l = len(seq)
    lengths.append(l)
    masks[i, :, l:] = False
    data.append((pdf_id, seq))
    i += 1

print(max(lengths))

print('getting embeddings...')
ids, batch_token, lengths = batchConverter(data)
model = load_prot_flash_small()
with torch.no_grad():
    token_embedding = model(batch_token, lengths)

# go from (N, L, d) to (N, d, L)
token_embedding = torch.transpose(token_embedding, 1, 2)

# option 1: N x M/2 x L
# generate masks
# masks = torch.full((2, 256, 71), True, dtype=torch.bool)
# l = len("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG")
# masks[0, :, l:] = False

# Build the mask vector
print('getting secondary structures...')
ss_model = Convolution_Predictor(512, masks)
output = torch.transpose(ss_model(token_embedding), 1, 2)
output_labels = torch.argmax(output, dim=2)
labels = {'H':0, 'B':1, 'E':2, 'G':3, 'I':4, 'T':5, 'S':6, 'L':7}
idx_to_label = {v:k for k, v in labels.items()}
# print(output_labels[0])
# final_label_0 = [idx_to_label[i] for i in output_labels[0].cpu().numpy()]
# string = ""
# print(string.join(final_label_0))
for l in output_labels:
    final = [idx_to_label[i] for i in l.cpu().numpy()]
    string = ""
    print(string.join(final))


# # Generate per-sequence representations via averaging
# sequence_representations = []
# for i, (_, seq) in enumerate(data):
#     sequence_representations.append(token_embedding[i, 0: len(seq) + 1].mean(0))

# model_data = torch.load(MODEL_URL_SMALL, map_location='cpu')
# hyper_parameter = model_data["hyper_parameters"]
# model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
#                              query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])
# model.load_state_dict(model_data['state_dict'])
# embeddings = model(batch_token, lengths)