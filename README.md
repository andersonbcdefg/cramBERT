# cramBERT
This repository contains my implementation of the BERT architecture and training, including the data pipeline to train on the OpenWebText2 dataset. I was inspired to carry out this project by recent work on efficient, low-budget BERT training. I came across the "Cramming paper" ([Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034)), and was impressed how much performance they could get out of a BERT that they trained for just one day on low-end hardware. They also managed to test lots of trendy Transformer architecture and training modifications along the way, and so their paper is the perfect guide to training a BERT. I took this paper as my starting point, supplementing with ideas I've found other places, to train a BERT that I can call my own. In my case, I'm training on a single A100 in Google Colab, which is a bit nicer and faster than what the authors had access to, but it supports a similar sort of training setup.

## Current Progress
So far, I've managed to train BERT with the MLM loss objective to a reasonable loss of around 1.9, which is solid and in line with the Cramming paper (though it can't be directly compared because I use a different tokenization scheme). Right now I'm running into some memory issues with data loaders, so I can only train on about 5 billion of the 10 billion tokens I set aside for training. After I can train on all 10 billion tokens and reach a reasonable pretraining loss, I'll evaluate the model on downstream tasks.

## Data and Preprocessing
I train and validate the model using the OpenWebText2 dataset, an open-source reproduction of OpenAI's WebText dataset by EleutherAI, which is a subset of the larger Pile dataset. (The Cramming authors experiment with several corpuses, including BookCorpus-Wikipedia, C4, and The Pile—I wanted to stick with just one solid dataset.) OpenWebText2 is comprised of over 17 million scraped documents, totaling around 65GB of uncompressed text. The dataset is available [here](https://github.com/EleutherAI/openwebtext2), along with preprocessing utilities. I directly borrowed some of this code to load the dataset from JSONL archives. It is licensed under the [MIT License](https://github.com/EleutherAI/openwebtext2/blob/master/LICENSE).

OpenWebText2 is already de-duplicated, so the only additional preprocessing steps I perform are filtering out documents that aren't in English, and filtering out documents that don't compress well when tokenized, as suggested by the Cramming paper. I experimented with some of the filtering techniques suggested by the creaters of the DeepMind Gopher model (Rae et al., 2022), but I found that they didn't remove many documents. I suspect that the full Common Crawl scrapes in the MassiveWeb dataset used to train Gopher contain more junk than OpenWebText2, which curates links from Reddit submissions. 

Unlike the authors of the Cramming paper, I do not convert the text to lowercase or strip out special characters and accents before tokenization. Every document from the OpenWebText2 dataset is left as-is.

## Tokenization
I further depart from the authors of the Cramming paper in my decision to use a byte-level BPE tokenizer, as I want the model to be able to represent any text on the web, including the special characters and accents I chose not to strip out. I use the HuggingFace Tokenizers library to train a BPE tokenizer from scratch on OpenWebText2, filtered to only English webpages. In the pre-tokenization step, the tokenizer applies NFC normalization, adds a prefix space to the start of each document (so that a word at the start of the document is considered the same as a word in the middle of a document), and splits on whitespace using the same regular expression as the GPT2 tokenizer. I use a vocabulary size of 32,768, the same size as the WordPiece vocabulary in the Cramming paper (and a [nice number that makes GPUs happy](https://twitter.com/karpathy/status/1621578354024677377)). After tokenization, all resulting tokens are packed into sequences of length 128, with documents separated by a special `[SEP]` token. This means no padding is required, and no computation is wasted on truncated sequences.


## Model Architecture
My implementation of BERT is quite similar to the [original paper]([https://arxiv](https://arxiv.org/abs/1810.04805)) by Devlin et al., with some tweaks suggested by more recent research. Some of these are identified in the Cramming paper, and many are now commonplace in the most recent wave of transformer models.
<details>
<summary>See Details</summary>

* I use the same 12-layer Transformer architecture as the original BERT paper, with 768-dimensional embeddings and 12 attention heads. As is now common, I place LayerNorms before, rather than after, the attention and feedforward sublayers, which improves training stability.
* For simplicity, I use learned absolute position embeddings. This means my model will not generalize beyond the sequence length used for training (128 tokens), but recent work on positional encoding (e.g. [Press, Smith, & Lewis, 2021](https://arxiv.org/abs/2108.12409)) finds that sinusoidal embeddings don't generalize well to longer sequences either.
* The feed-forward networks in my Transformer use the Gated Linear Units proposed by Noam Shazeer (2020) and suggested in the Cramming paper. I reduce the hidden size to 2,048 (rather than 3,072) to maintain the same number of parameters.
* I omit biases for all feed-forward layers, including the query-key-value projections in the attention sublayer. I also omit the bias in the affine transformations that follow LayerNorms. Omitting bias is a common practice in recent Transformer models, and is suggested in the Cramming paper as a way to simplify and speed up training, without measurably reducing the *size* of the model (which tends to hurt performance).
* Weights in all linear layers are initialized randomly from a normal distribution with mean 0 and standard deviation 0.002. I found that a standard deviation of 0.02 (as in OpenAI's GPT-2) resulted in a large initial loss, indicating that a smaller initialization would be better. I'm sure that Kaiming or Xavier uniform initialization would work fine too, the important thing seemed to be making sure the weights were *small* enough. Positional embeddings were initialized to 0, and the LayerNorm weights were initialized to 1.
* For token embeddings, I use the StableEmbedding module from the `bitsandbytes` library, which is a drop-in replacement for `torch.nn.Embedding` that is more stable when using an 8-bit optimizer. It includes a LayerNorm, so I do not need to add my own LayerNorm directly after the token embedding. I add an additional LayerNorm after adding the positional embedding to the token embedding.

</details>

## Training Details
I'm still working on training as of February 2023, but here's the details so far:
* Sequence length of 128, which saves in memory. The original BERT was trained on 128-length sequences for 90% of its training, and recent work on low-budget BERT training suggests that 128-length sequences are sufficient for most tasks.
* Model trained on OpenWebText2, using approximately 10 billion tokens for training, and 1% of that for validation.
* Optimized with AdamW, weight decay applied to all parameters except bias and LayerNorm weights.
* Gradient clipping to stabilize training (clip norm 0.5).
* Ramp batch size throughout training as suggested by Cramming paper. The largest batch size that fits on a single GPU is 256, and I use gradient accumulation to ramp the batch size up to 4,096 sequences by the end of training. (Recommended size by Izsak et al. and Cramming paper)
* Gradient accumulation used for an effective batch size of 4,096 sequences (seems to be generally recommended/good batch size for pretrained language models), found best by Izsak et al.
* Mixed precision training using torch.cuda.amp autocast and gradient scaler.
* Save memory with 8-bit Adam/AdamW from `bitsandbytes`, using StableEmbeddings layer from `bitsandbytes` to keep training stable.
* The learning rates and linear one-cycle schedule along the lines of those recommended in Cramming and Iszak papers did not lead to stable training in my setting—I suspect that OpenWebText2 with BPE may lead to a noisier dataset, which must be compensated for with a larger batch size or a smaller learning rate. (OpenAI's work on the [gradient noise scale](https://openai.com/blog/science-of-ai/) argues that more-difficult tasks have noisier gradients, and that this largely determines the optimal batch size). I found that a lower maximum learning rate of 1e-4, with warmup for X% of the training data and cosine annealing for the rest of training worked well for me.

## Results
So far, I've achieved a MLM loss of around 1.9! I plan to fine-tune and evaluate the model on a few downstream tasks to gauge how well it performs there. I'll update this section as I make progress.


## References
* "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)", Devlin et al., 2019
* "[How to Train BERT with an Academic Budget](https://arxiv.org/abs/2104.07705)", by Peter Izsak, Moshe Berchansky, and Omer Levy
* "[Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/pdf/2212.14034.pdf)", by Jonas Geiping and Tom Goldstein
* "[Scaling Language Models: Methods, Analysis & Insights from Training Gopher"](https://arxiv.org/abs/2112.11446), by Jack Rae et al.
* 

# Acknowledgments
Special thanks to Jonas Geiping, an author of the Cramming paper who was exceptionally helpful and kind in answering my questions about the paper, code, and training details.