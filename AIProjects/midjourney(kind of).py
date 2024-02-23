import numpy as np
import torch
import clip
# from pkg_resources import packaging
print("Torch version:", torch.__version__)


print('Available models:')
print(clip.available_models())

model, preprocess = clip.load("ViT-L/14") # used by stable diffusion v1
# model.cuda().eval()
# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size
#
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)

prompt = input('Your prompt: ')
tokens = clip.tokenize(prompt)
# with torch.no_grad():
#     embeddings = model.token_embedding(tokens).float()
print("text tokens:")
print(tokens)
# print("text tokens size:", tokens.shape)
# print("Embeddings size:", embeddings.shape)








