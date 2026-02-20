# Shakespeare GPT

A character-level language model trained on Shakespeare's writings, built from scratch using PyTorch.

## What it does

This project trains a small GPT-style transformer to generate new text in the style of Shakespeare. Feed it ~1.1 million characters of Shakespeare, and it learns to write like him — iambic prose, thees, thous, and all.

## How it works

The model is a decoder-only transformer with:
- 6 transformer blocks, each with 6 attention heads
- 384-dimensional embeddings
- 256-token context window
- ~10M parameters total

It's trained character-by-character, so it learns everything from scratch — no pretrained weights, no tokenizer.

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
```

On Windows:
```bash
.venv\Scripts\activate
```

On Mac/Linux:
```bash
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

A GPU is recommended but not required — the model automatically falls back to CPU.

## Usage

```bash
python bigram.py
```

On first run, it trains for 5000 steps and saves a checkpoint (`bigram_checkpoint.pt`). On subsequent runs it loads the checkpoint and jumps straight to generation.

## Training data

The [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset — a collection of Shakespeare's works concatenated into a single text file (~1.1M characters). Download it and save it as `input.txt` in the project directory.

## Sample output

```
KING RICHARD:
What art thou, that dost make thy sorrow known,
And yet wouldst have me think thee innocent?
```
*(actual output will vary each run)*
