# GPT1-numpy

A tiny, from-scratch NumPy implementation of an OpenAI GPT-1 style language model.
This project loads a pretrained GPT-1 checkpoint, runs a pure-NumPy forward pass,
and generates text with a minimal custom tokenizer.

The goal is to keep the code short and readable while showing the full inference path:

- Custom BPE-ish tokenizer (no external tokenizer library)
- Token + position embeddings
- Multi-head self-attention
- MLP blocks
- Post-LayerNorm GPT-1 block order
- Simple top-p sampling with temperature and repetition penalty

## Project layout

- [inference.py](inference.py): single-file inference script
- [model/](model/): GPT-1 checkpoint and tokenizer assets
  - config.json
  - merges.txt
  - model.safetensors
  - tokenizer.json
  - vocab.json

## Requirements

- Python 3.10+
- numpy

Install deps in your venv:

```bash
pip install numpy
```

## Run

```bash
python .\inference.py
```

The script prints each generated token and then the final combined text.

## Notes

- The tokenizer is intentionally small and custom. It uses the project
  tokenizer.json + merges.txt and a simple pre-tokenizer to avoid merging
  across word boundaries.
- This is **inference-only**. There is no training or backprop here.
- The checkpoint is loaded from [model/](model/). Keep those files together.

## Tuning generation

In [inference.py](inference.py), you can adjust:

- `temperature`
- `top_p`
- `repetition_penalty`
- `prompt`

## Why this exists

This is a learning-first, minimal GPT-1 inference walkthrough. The goal is to
mirror the core model math without hiding details behind large frameworks.

## License

Use it for learning and personal experiments. If you redistribute the model
files, make sure you have the right to do so.
