"""
By default, the optimizer parameters are also saved with the model.
"""
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('model_path')
args = parser.parse_args()

ckpt = torch.load(args.model_path)
torch.save(dict(
    state_dict=ckpt['state_dict'],
    hyper_parameters=ckpt['hyper_parameters'],
), args.model_path.replace('.ckpt', '.compact.ckpt'))
