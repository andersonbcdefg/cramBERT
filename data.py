import os
import glob
import torch
import math
import requests
import pathlib
from tqdm.auto import tqdm
import tarfile
import numpy as np
import random
import transformers
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from webtext.archiver import Reader
from itertools import chain, filterfalse
from functional import seq
from collections import Counter
from multiprocessing import Pool
from mmap import mmap, ACCESS_READ

WEBTEXT_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar"
WEBTEXT_RAW_PATH = "webtext/openwebtext2.jsonl.zst.tar"
WEBTEXT_EXTRACTED_FOLDER = "webtext/extracted"
BILLION = math.pow(10, 9)
NUM_ENGLISH_WEBTEXT_DOCS = 13570754

def download_webtext(file_path=WEBTEXT_RAW_PATH, url=WEBTEXT_URL, 
                    extracted_folder=WEBTEXT_EXTRACTED_FOLDER, force=False):
    print(f"Downloading webtext dataset from {url}...")
    webtext = pathlib.Path(file_path)
    if webtext.exists() and not force:
        print("Already downloaded, skipping download.")
    else:
        res = requests.get(WEBTEXT_URL, stream=True)
        if res.status_code != 200:
            raise Exception("Could not download webtext dataset.")
        total_size_in_bytes = int(res.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm.tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(webtext, "wb") as f:
            for data in res.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
    if not pathlib.Path(extracted_folder).exists():
        # make the extracted folder
        pathlib.Path(extracted_folder).mkdir(parents=True, exist_ok=True)
    if len(glob.glob(extracted_folder + "/*jsonl.zst")) > 0 and not force:
        print("Already extracted, skipping extraction.")
        return
    print("Extracting...")
    with tarfile.open(webtext, "r") as tar:
        tar.extractall(extracted_folder)
    print("Successfully downloaded and extracted a total of " +\
            len(glob.glob(extracted_folder + "/*jsonl.zst")) +\
             " files.")

"""
Simple iterator over the WebText dataset.
"""
def webtext_batch_iterator(extracted_dir=WEBTEXT_EXTRACTED_FOLDER, batch_size=1000, 
                            english_only=True, shuffle_files=False):
    files = glob.glob(extracted_dir + "/*jsonl.zst")
    if shuffle_files:
        random.shuffle(files)
    for file_path in files:
        reader = Reader()
        documents = [{
            "document": document, 
            "metadata": metadata
        } for document, metadata in reader.read_jsonl(file_path, get_meta=True)]
        if english_only:
            documents = [doc for doc in documents if doc["metadata"]["lang"] == "en"]
        documents = [doc["document"] for doc in documents]
        for i in range(0, len(documents), batch_size):
            yield documents[i:i+batch_size]

"""
Get information about OpenWebText2 dataset (number of documents, lengths, etc.)
"""
def count_webtext_english_docs(extracted_dir=WEBTEXT_EXTRACTED_FOLDER):
    pool = Pool(8)
    files = glob.glob(extracted_dir + "/*jsonl.zst")
    num_documents = 0
    reader = Reader()
    for file_path in tqdm.tqdm(files):
        documents = [{
            "document": document, 
            "metadata": metadata
        } for document, metadata in reader.read_jsonl(file_path, get_meta=True)]
        num_english = len([doc for doc in documents if doc["metadata"]["lang"] == "en"])
        num_documents += num_english
    print(f"Total number of English documents: {num_documents}")

"""
Train a tokenizer on the WebText dataset.
See: https://huggingface.co/docs/tokenizers/quicktour
"""
def train_tokenizer(file_path="webtext/tokenizer.json", extracted_dir=WEBTEXT_EXTRACTED_FOLDER, force=False):
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFC() # Relatively lossless normalization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    if pathlib.Path(file_path).exists() and not force:
        print("Tokenizer already exists at that path! Use force=True to overwrite.")
    else:
        print("Training tokenizer...")
        trainer = trainers.BpeTrainer(
            vocab_size=32768,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[PAD]", "[MASK]", "[SEP]"]
        )
        tokenizer.train_from_iterator(
            webtext_batch_iterator(extracted_dir=extracted_dir, english_only=True), 
            trainer=trainer, 
            length=NUM_ENGLISH_WEBTEXT_DOCS
        )
        print("Saving tokenizer to file...")
        tokenizer.save(file_path)
    return tokenizer

def load_tokenizer(file_path="webtext/tokenizer.json"):
    plain_tokenizer = Tokenizer.from_file(file_path)
    fast_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=plain_tokenizer)
    # add special tokens
    fast_tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]", "sep_token": "[SEP]"})
    return fast_tokenizer


"""
Use tokenizer to encode a batch of documents, applying heuristic filters
to remove documents likely to negatively impact training. EleutherAI already
filters out duplicates with MinHash, and the longest document in OpenWebText2
is only 100,000 characters long, which is fine. We remove non-English documents
before passing them to this function, so only add two filters here:
- Remove documents that are too short or repetitive (< 50 unique words) before tokenizing
- Remove non-compressible documents after tokenizing (tokens must be < 30% of original length in chars)
"""
def filter_and_batch_encode(documents, tokenizer=None):
    # Pre-tokenization filters
    documents = [doc for doc in documents if len(set(doc.split())) >= 50]
    if tokenizer is None:
        return documents

    # Tokenization & filtering non-compressible documents
    encoded_docs = tokenizer(documents, add_special_tokens=False).input_ids
    encoded_docs = [doc for i, doc in enumerate(encoded_docs) if len(doc) < 0.3 * len(documents[i])]
    return encoded_docs

"""
Get a subset of the WebText dataset to train/validate on.
Returns 2-dimensional array of tokens, packed into 128-length sequences.
See: https://openwebtext2.readthedocs.io/en/latest/
"""
def load_and_prep_webtext(train_tokens=10 * BILLION, val_frac=0.01, max_seq_len=128, 
                            train_npy_file="webtext/webtext_train.bin", val_npy_file="webtext/webtext_val.bin", force=False):
    tokenizer = load_tokenizer(file_path="webtext/tokenizer.json")
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    train_file = open(train_npy_file, "wb+")
    val_file = open(val_npy_file, "wb+")
    total_tokens_needed = train_tokens + int(val_frac * train_tokens)
    total_tokens = 0
    document_count = 0
    with tqdm.tqdm(total = total_tokens_needed) as pbar:
        for batch in webtext_batch_iterator(extracted_dir=WEBTEXT_EXTRACTED_FOLDER, 
            batch_size=500, shuffle_files=True, english_only=True):
            document_count += len(batch)
            encodings = filter_and_batch_encode(batch, tokenizer=tokenizer)
            new_tokens = np.fromiter(chain(*[encoding + [sep_token_id] for encoding in encodings]), dtype=np.uint16)
            total_tokens += len(new_tokens)
            if total_tokens < total_tokens_needed:
                if random.random() < val_frac:
                    new_tokens.tofile(val_file)
                else:
                    new_tokens.tofile(train_file)
            else:
                break
            pbar.update(len(new_tokens))
    train_file.close()
    val_file.close()
    train_tokens = np.memmap(train_npy_file, dtype=np.uint16, mode="r")
    val_tokens = np.memmap(val_npy_file, dtype=np.uint16, mode="r")
    
    # print results
    print(f"Loaded {document_count} documents from OpenWebText2.")
    print(f"Saved {len(train_tokens) / math.pow(10, 6)}m tokens to {train_npy_file}.")
    print(f"Saved {len(val_tokens) / math.pow(10, 6)}m tokens to {val_npy_file}.")
    print("Done!")

# def get_masked_tokens(tokens, vocab_size, mask_token_id, mask_prob=0.15, 
#                         random_prob=0.1, orig_prob=0.1):
#     mask = np.random.choice([0, 1], size=tokens.shape, p=[1 - mask_prob, mask_prob])
#     mask_variations = np.random.choice([0, 1, 2], size=tokens.shape, p=[1 - random_prob - orig_prob, random_prob, orig_prob])
#     random_tokens = np.random.randint(vocab_size, size=tokens.shape)

#     # 0: regular mask, 1: random token, 2: original token
#     masked_tokens = np.where(np.logical_and(mask == 1, mask_variations == 0), mask_token_id, tokens)
#     masked_tokens = np.where(np.logical_and(mask == 1, mask_variations == 1), random_tokens, masked_tokens)
#     return masked_tokens, mask


# If debug is True, will return original tokens, along with masked tokens and target.
class BERTDataset(torch.utils.data.IterableDataset):
    def __init__(self, raw_data_path, tokenizer, seq_len, mask_prob=0.15, max_seqs=0, loop=False, debug=False):
        super().__init__()

        # Params for mmap to raw data file
        self.raw_data_path = raw_data_path
        self.bytes_per_seq = seq_len * 2
        self.usable_bytes = os.path.getsize(raw_data_path) // self.bytes_per_seq * self.bytes_per_seq
        self.n_seqs = self.usable_bytes // self.bytes_per_seq
        if max_seqs > 0:
            self.n_seqs = min(self.n_seqs, max_seqs)
        self.loop = loop
        self.debug = debug
        print(f"Loading {self.n_seqs} sequences of length {seq_len} from {raw_data_path}.")
        
        # Collator
        self.collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mask_prob)
        
    def mmap_iterator(self, start_seq, end_seq):
        raw_data_file = open(self.raw_data_path, "r+b")
        mm = mmap(raw_data_file.fileno(), 0, access=ACCESS_READ)
        start_pos, end_pos = start_seq * self.bytes_per_seq, end_seq * self.bytes_per_seq
        mm.seek(start_pos)
        while True:
            # If we've reached end of assigned range, reset to start of range
            if mm.tell() >= end_pos:
                if self.loop:
                    mm.seek(start_pos)
                else:
                    return
            seq_bytes = mm.read(self.bytes_per_seq)
            np_inputs = np.frombuffer(seq_bytes, dtype=np.uint16).astype(np.int64)
            as_tensor = torch.LongTensor(np_inputs)
            if self.debug:
                orig_inputs = as_tensor.clone()
            inputs, targets = self.collator.torch_mask_tokens(as_tensor.reshape(1, -1))
            if self.debug:
                yield (
                    inputs.reshape(-1),
                    targets.reshape(-1),
                    orig_inputs.reshape(-1)
                )
            else:
                yield (
                    inputs.reshape(-1), 
                    targets.reshape(-1)
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.n_seqs
            print(f"Single process loading sequences {iter_start} to {iter_end - 1}")
        else:  # in a worker process, split workload
            seqs_per_worker = int(math.ceil(self.n_seqs / worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * seqs_per_worker
            iter_end = min(iter_start + seqs_per_worker, self.n_seqs)
            print(f"Worker {worker_id} assigned sequences {iter_start} to {iter_end}")
        return iter(self.mmap_iterator(iter_start, iter_end))

    def get_random_seq(self, masked=False):
        raw_data_file = open(self.raw_data_path, "r+b")
        mm = mmap(raw_data_file.fileno(), 0, access=ACCESS_READ)
        seq_pos = random.randint(0, self.n_seqs - 1) * self.bytes_per_seq
        mm.seek(seq_pos)
        seq_bytes = mm.read(self.bytes_per_seq)
        np_inputs = np.frombuffer(seq_bytes, dtype=np.uint16).astype(np.int64)
        seq = torch.LongTensor(np_inputs)
        if masked:
            seq = self.collator.torch_mask_tokens(seq.reshape(1, -1))[0].reshape(-1)
        return seq

if __name__ == "__main__":
    download_webtext()
    load_and_prep_webtext(train_tokens=10 * BILLION)
    