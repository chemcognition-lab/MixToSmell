from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import logging
import numpy as np
import torch
from typing import List

UNK_TOKEN = -999


# def pad_list(data, pad_value=UNK_TOKEN, max_length=None):
#     if not isinstance(data, list):
#         return data  # Return as-is if it's not a list
#     if max_length is None:
#         max_length = len(data)  # Default to current length if max_length isn't set
#     return data + [pad_value] * (max_length - len(data))

def pad_list(data, pad_value=UNK_TOKEN, max_length=None, max_inner_length=None):
    if not isinstance(data, list):
        return [pad_value] * (max_length or 1)

    # If it's a list of lists (nested)
    if all(isinstance(item, list) for item in data):
        # Determine max lengths if not provided
        if max_inner_length is None:
            max_inner_length = max((len(sublist) for sublist in data), default=0)
        if max_length is None:
            max_length = len(data)

        # Pad inner lists
        padded_inner = [
            sublist + [pad_value] * (max_inner_length - len(sublist)) 
            for sublist in data
        ]
        # Pad outer list
        padded_outer = padded_inner + [[pad_value] * max_inner_length] * (max_length - len(data))
        return padded_outer

    # Otherwise, treat it as a flat list
    if max_length is None:
        max_length = len(data)
    return data + [pad_value] * (max_length - len(data))


def indices_to_graphs(data_list, index_tensor):
    selected = []
    for idx in index_tensor:
        if not idx == UNK_TOKEN:
            selected.append(data_list[int(idx.item())].clone())
        else:
            selected.append(UNK_TOKEN)
    return selected