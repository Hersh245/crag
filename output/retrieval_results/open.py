import pickle as pkl
import numpy as np
import torch

with open('./example_data.pkl', 'rb') as f:
    dict = pkl.load(f)
relevances = sorted(dict['3dbed55e-66a3-4dcd-907d-096f49387e41'][0], reverse=True)
for score in relevances[:50]:
    print(round(score, 5))