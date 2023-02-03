# cramBERT
My implementation of BERT, along with code to train the model along the lines of previous researchers working with limited resources.

# Implementation / Training Details
* Unlike BERT, I use byte-level BPE to tokenize the data, because I want to be able to represent any sequence from the web, and avoid stripping out special characters and accents as done in the Cramming paper. I keep the same vocabulary size as the Cramming paper (32768), which is a nice block size for GPU training (it's divisible by 64, 128, etc.). I train the BPE from scratch on OpenWebText2.
* Model trained on OpenWebText2, using approximately 10 billion tokens for training, and 1% of that for validation.
* Two-phase, one-cycle learning rate schedule, as recommended in the Cramming paper.
* Optimized with AdamW, weight decay applied to all parameters except bias and LayerNorm weights.
* Gradient clipping to stabilize training (clip norm at 1.0 or 0.5?).
* Gradient accumulation used for an effective batch size of 4,096 sequences (seems to be generally recommended/good batch size for pretrained language models).
* Sequence length of 128 tokens

Based on the following papers:
* "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)", Devlin et al., 2019
* "[How to Train BERT with an Academic Budget](https://arxiv.org/abs/2104.07705)", by Peter Izsak, Moshe Berchansky, and Omer Levy
* "[Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/pdf/2212.14034.pdf)", by Jonas Geiping and Tom Goldstein

The OpenWebText2 dataset used to train the model was collected by EleutherAI and is available [here](https://github.com/EleutherAI/openwebtext2), along with preprocessing utilities. I directly borrowed some of this code to load the dataset from JSONL archives. It is licensed under the [MIT License](https://github.com/EleutherAI/openwebtext2/blob/master/LICENSE).