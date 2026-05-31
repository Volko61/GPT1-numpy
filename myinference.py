# 1 find the files                          
import os
import json
from typing import List
import numpy as np
from safetensors.numpy import load_file

path = "C:\\Users\\volko\\.cache\\huggingface\\hub\\models--openai-gpt\\snapshots\\1e0d4f3028acbffb47fe933cea64619c5ec1a002"

with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

vocab_size = config["vocab_size"]       
n_positions = config["n_positions"]     
n_embds = config["n_embd"]              
n_layers = config["n_layer"]              
n_heads = config["n_head"]   
head_dim = n_embds // n_heads 

# 2 tokenize                                    -> text to numbers
def merge(sentence: str) -> List[str]:
    result = list(sentence)
    with open(os.path.join(path, "merges.txt"), "r", encoding="utf-8") as f:
        for rule in f.readlines()[1:]:
            rule = rule[:-1].split(" ")
            # print("rule :", rule)
            i = 0
            while i < len(result) - 1:
                a = result[i]
                b = result[i + 1]
                if a == rule[0] and b == rule[1]:
                    # print("merging:", a, "+", b)
                    result[i:i+2] = [rule[0] + rule[1]]
                else:
                    i += 1
    return result


def tokens_to_tensor(tokens: str) -> List[int]:
    with open(os.path.join(path, "tokenizer.json"), "r", encoding="utf-8") as f:
        rules = json.load(f)["model"]["vocab"]
        # print(rules)
        result = []
        for token in tokens:
            result.append(rules[token])

    return result

def tensor_to_token(tensors: List[int]) -> str:
    with open(os.path.join(path, "tokenizer.json"), "r", encoding="utf-8") as f:
        rules = json.load(f)["model"]["vocab"]
        # print(rules)
        result = ""
        for tensor in tensors:
            for key, value in rules.items():
                if value == tensor:
                    result += key

    return result

# print(tokens_to_tensor(merge("there")))

# 3 embedding                                   -> numbers to ideas
# Load the weights into a dictionary of numpy arrays
weights = load_file(os.path.join(path, "model.safetensors"))
def embedding(token_ids: List[int]) -> np.ndarray:
    word_token_embeddings = weights["tokens_embed.weight"]
    word_position_embeddings = weights["positions_embed.weight"]
    
    result = []
    for i, token in enumerate(token_ids):
        word_token_embedding = word_token_embeddings[token]
        word_position_embedding = word_position_embeddings[i]
        result.append(word_token_embedding + word_position_embedding)
    return np.array(result)


def lm_head(hidden_states):
    return hidden_states @ weights["tokens_embed.weight"].T

# print(embedding([5124]))

# 4 transformer loop (matrix multiplications)   -> guess new idea probabilities

def layer_norm(x, weight, bias, eps=1e-5):
    mean= np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) /np.sqrt(var +eps) + bias

def softmax(x):
    probs = np.exp(x - np.max(x, axis=-1, keepdims=True))
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return probs

def mha_block(x, l):
    seq_len = x.shape[0]

    #QVK
    qkv = x @ weights[f"h.{l}.attn.c_attn.weight"] + weights[f"h.{l}.attn.c_attn.bias"]
    q, k, v = np.split(qkv, 3, axis=-1)

    q = q.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
    k = k.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)
    v = v.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)

    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(head_dim)

    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 1, scores, -1e9)

    probs = softmax(scores)

    # combine heads back together
    context = (probs @ v).swapaxes(0, 1).reshape(seq_len, n_embds)

    # projection 
    out = context @ weights[f"h.{l}.attn.c_proj.weight"] + weights[f"h.{l}.attn.c_proj.bias"]
    return out

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def mlp_block(x, l):
    h1 = gelu(x @ weights[f"h.{l}.mlp.c_fc.weight"] + weights[f"h.{l}.mlp.c_fc.bias"] )
    out = h1 @ weights[f"h.{l}.mlp.c_proj.weight"] + weights[f"h.{l}.mlp.c_proj.bias"]
    return out

def transformer_loop(hidden_states):
    for l in range(n_layers):
        # Attention
        ln1_x = layer_norm(hidden_states, weights[f"h.{l}.ln_1.weight"], weights[f"h.{l}.ln_1.bias"])
        hidden_states = hidden_states + mha_block(ln1_x, l)
        # MLP
        ln2_x = layer_norm(hidden_states, weights[f"h.{l}.ln_2.weight"], weights[f"h.{l}.ln_2.bias"])
        hidden_states = hidden_states + mlp_block(ln2_x, l)

    return hidden_states

prompt = "hi"
for i in range(10):
    hidden_states = transformer_loop(embedding(tokens_to_tensor(merge(prompt.replace(" ", "</w>")))))

    # 5 projection                                  -> new idea probablities to number probabilities
    logits = lm_head(hidden_states)

    # 6 sampling                                    -> select the most probable number and convert back to text thanks to tokenizer
    prompt += tensor_to_token([logits[-1].argmax()])

print(prompt)





























