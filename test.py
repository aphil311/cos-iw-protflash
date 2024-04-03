from ProtFlash.pretrain import load_prot_flash_small
from ProtFlash.utils import batchConverter
import torch

from ProtFlash.model import FLASHTransformer


MODEL_URL_SMALL = "/scratch/network/ap9884/flash_protein.pt"

data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
]
ids, batch_token, lengths = batchConverter(data)
model = load_prot_flash_small()
with torch.no_grad():
    token_embedding = model(batch_token, lengths)
# Generate per-sequence representations via averaging
sequence_representations = []
for i, (_, seq) in enumerate(data):
    sequence_representations.append(token_embedding[i, 0: len(seq) + 1].mean(0))

model_data = torch.load(MODEL_URL_SMALL, map_location='cpu')
hyper_parameter = model_data["hyper_parameters"]
model = FLASHTransformer(hyper_parameter['dim'], hyper_parameter['num_tokens'], hyper_parameter['num_layers'], group_size=hyper_parameter['num_tokens'],
                             query_key_dim=hyper_parameter['qk_dim'], max_rel_dist=hyper_parameter['max_rel_dist'], expansion_factor=hyper_parameter['expansion_factor'])

model.load_state_dict(model_data['state_dict'])
print(model(batch_token, lengths))
