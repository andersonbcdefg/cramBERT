# cramBERT
My implementation of BERT, along with code to train the model along the lines of previous researchers working with limited resources. This is a work in progress, and I will update this README as I make progress.

## Data and Preprocessing
I train and validate the model using the OpenWebText2 dataset, an open-source reproduction of OpenAI's WebText dataset by EleutherAI. OpenWebText2 comprises over 17 million scraped documents, totaling around 65GB of uncompressed text. The dataset is available [here](https://github.com/EleutherAI/openwebtext2), along with preprocessing utilities. I directly borrowed some of this code to load the dataset from JSONL archives. It is licensed under the [MIT License](https://github.com/EleutherAI/openwebtext2/blob/master/LICENSE).

OpenWebText2 is already de-duplicated. The only other preprocessing steps I perform are to filter out documents not written in English (using the provided metadata), and filtering out documents that don't compress well when tokenized, as suggested by the Cramming paper (Geiping and Goldstein, 2021). This would likely remove things like DNA sequences, binary or base64 encodings, and other things of that nature. I experimented with some of the techniques suggested in the Gopher paper (Rae et al., 2022) for filtering data scraped from the web, but I found that they removed negligible fractions of documents from the OpenWebText2 corpus—I suspect that raw scrapes from Common Crawl require much more filtering than documents from Reddit submissions, which is how the WebText methodology collects webpages. I do not strip out special characters or accents before tokenization—documents are left as-is.

## Tokenization
I depart from the authors of the Cramming paper in my decision to use a byte-level BPE tokenizer, as I want the model to be able to represent any text on the web, including the special characters and accents I chose not to strip out. Using the same vocabulary size of 32,768 (which is a [nice number that makes GPUs happy](https://twitter.com/karpathy/status/1621578354024677377)) I train the BPE tokenizer from scratch on OpenWebText2, filtered to only English webpages. I use the HuggingFace Tokenizers library. In the pre-tokenization step, the tokenizer applies NFC normalization, adds a prefix space to the start of each document (so that a word at the start of the document is considered the same as a word in the middle of a document), and splits on whitespace using the same regular expression as the GPT2 tokenizer.


## Model Architecture
My implementation of BERT is quite similar to the original paper by Devlin et al., with some tweaks suggested by more recent research, especially the Cramming paper (but a lot of these are widely known/used now).
* I use the same 12-layer Transformer architecture as the original BERT paper, with 768 hidden units, 12 attention heads.
* For simplicity, I use learned absolute position embeddings, as they aren't much worse or better than sinusoidal ones, and are trivial to implement. This means my model will not generalize beyond the sequence length used for training (128 tokens), but that's true with sinusoidal embeddings too.
* FFNs use gating (GEGLU) as in Shazeer's work and suggest by Cramming paper, and I reduce the FFN hidden size to 2048 to maintain roughly the same number of parameters.
* I use pre-LN for stability, as suggested by RoBERTa, Cramming paper and like literally everyone at this point.
* FFNs do not have bias, and LayerNorms do not have bias either.
* Tie weights in embedding layer and output layer.

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
* The learning rates and linear one-cycle schedule along the lines of those recommended in Cramming and Iszak papers did not lead to stable training in my setting—I suspect that OpenWebText2 with BPE may lead to a noisier dataset, which must be compensated for with a larger batch size or a smaller learning rate. (OpenAI's work on the gradient noise scale argues that more-difficult tasks have noisier gradients, and that this largely determines the optimal batch size). I found that a lower maximum learning rate of 1e-4, with warmup for X% of the training data and cosine annealing for the rest of training worked well for me.

## Downstream Performance
I plan to fine-tune and evaluate the model on a few downstream tasks to gauge how well it performs. But I have to get it to a good MLM loss first!


References
* "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)", Devlin et al., 2019
* "[How to Train BERT with an Academic Budget](https://arxiv.org/abs/2104.07705)", by Peter Izsak, Moshe Berchansky, and Omer Levy
* "[Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/pdf/2212.14034.pdf)", by Jonas Geiping and Tom Goldstein
* "[Scaling Language Models: Methods, Analysis & Insights from Training Gopher"](https://arxiv.org/abs/2112.11446), by Jack Rae et al.
* 

# Acknowledgments
Special thanks to Jonas Geiping, an author of the Cramming paper who was exceptionally helpful and kind in answering my questions about the paper, code, and training details.