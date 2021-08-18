# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 07:05:16 2021

@author: user
"""

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print( tokenizer.tokenize("This is a sample text to test the tokenizer.") )
tokens    = tokenizer.tokenize("This is a sample text to test the tokenizer.")
print( tokenizer.convert_tokens_to_ids( tokens ) )

import torch
from transformers import AutoModel
pretrained_model_name = "bert-base-uncased"
tokenized_input = tokenizer.encode("This is a sample text to test the tokenizer.")
model = AutoModel.from_pretrained(pretrained_model_name)

output = model( torch.tensor([tokenized_input]) )

print( output[0].size(), output[1].size() )