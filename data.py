import glob
import torch
import math
import requests
import pathlib
import tqdm
import tarfile
import numpy as np
import random
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from webtext.archiver import Reader

WEBTEXT_URL = "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar"
WEBTEXT_RAW_PATH = "webtext/openwebtext2.jsonl.zst.tar"
WEBTEXT_EXTRACTED_FOLDER = "webtext/extracted"
BILLION = math.pow(10, 9)

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
Simple iterator over the WebText dataset, used to train the tokenizer.
"""
def webtext_batch_iterator(extracted_dir=WEBTEXT_EXTRACTED_FOLDER, batch_size=1000):
    files = glob.glob(extracted_dir + "/*jsonl.zst")
    for file_path in files:
        reader = Reader()
        documents = [document for document, metadata in reader.read_jsonl(file_path, get_meta=True)]
        for i in range(0, len(documents), batch_size):
            yield documents[i:i+batch_size]

def train_or_load_tokenizer(file_path="webtext/tokenizer.json", extracted_dir=WEBTEXT_EXTRACTED_FOLDER, force=False):
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    if pathlib.Path(file_path).exists() and not force:
        print("Loading tokenizer from file...")
        tokenizer = Tokenizer.from_file(file_path)
    else:
        print("Training tokenizer...")
        trainer = trainers.BpeTrainer(
            vocab_size=32768,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[PAD]", "[MASK]", "[SEP]"]
        )
        tokenizer.train_from_iterator(webtext_batch_iterator(extracted_dir=extracted_dir), trainer=trainer)
        print("Saving tokenizer to file...")
        tokenizer.save(file_path)
    return tokenizer


"""
Get a subset of the WebText dataset to train on.
Returns 2-dimensional array of tokens, packed into 128-length sequences.
See: https://openwebtext2.readthedocs.io/en/latest/
"""
def load_and_prep_webtext(randomize=True, max_train_tokens=5 * BILLION, val_frac=0.01, max_seq_len=128, 
                            train_npy_file="webtext/webtext_train.npy", val_npy_file="webtext/webtext_val.npy", force=False):
    tokenizer = train_or_load_tokenizer()
    train_file = pathlib.Path(train_npy_file)
    val_file = pathlib.Path(val_npy_file)
    files = glob.glob(WEBTEXT_EXTRACTED_FOLDER + "/*jsonl.zst")
    if randomize:
        random.shuffle(files)
    total_tokens_needed = max_train_tokens * (1 + val_frac)
    total_tokens = 0
    document_count = 0
    tokens = []
    with tqdm.tqdm(total = total_tokens_needed) as pbar:
        for file_path in files:
            reader = Reader()
            for document, metadata in reader.read_jsonl(file_path, get_meta=True):
                if total_tokens > tokens_needed:
                    break
                document_count += 1
                doc_tokens = tokenizer.encode(document, add_special_tokens=False)
                if len(tokens) > 0:
                    doc_tokens.append(SEP_TOKEN_ID)
                    tokens.extend(doc_tokens)
                else: 
                    tokens = doc_tokens
                total_tokens += len(doc_tokens)
                pbar.update(len(doc_tokens))
    n_all_seqs = math.floor(len(tokens) / max_seq_len)
    trunc_tokens = tokens[:n_all_seqs * max_seq_len]
    all_seqs = np.array(trunc_tokens).reshape(-1, max_seq_len)
    np.random.shuffle(all_seqs)

    # split into train and val
    num_train_seqs = math.floor(len(all_seqs) * (1 - val_frac))
    train_seqs = all_seqs[:num_train_seqs]
    val_seqs = all_seqs[num_train_seqs:]

    # save to disk
    np.save(train_npy_file, train_seqs)
    np.save(val_npy_file, val_seqs)
    
    # print results
    print(f"Loaded {document_count} documents, containing {total_tokens} tokens.")
    print(f"Saved {len(train_seqs)} train sequences to {train_npy_file}.")
    print(f"Saved {len(val_seqs)} val sequences to {val_npy_file}. Done!")
    return train_seqs, val_seqs

def get_masked_tokens(tokens, vocab_size, mask_prob=0.15, 
                        random_prob=0.1, orig_prob=0.1):
    n_examples = len(tokens)
    seq_len = len(tokens[0])
    print("Generating masks...")
    mask = np.random.choice([0, 1], size=(n_examples, seq_len), p=[1 - mask_prob, mask_prob])
    print("Generating mask variations...")
    mask_variations = np.random.choice([0, 1, 2], size=(n_examples, seq_len), p=[1 - random_prob - orig_prob, random_prob, orig_prob])
    print("Generating random tokens...")
    random_tokens = np.random.randint(vocab_size, size=(n_examples, seq_len))
    print("Masking tokens...")
    masked_tokens = np.copy(tokens)
    masked_tokens = np.where(mask == 1, MASK_TOKEN_ID, masked_tokens)
    masked_tokens = np.where(np.logical_and(mask == 1, mask_variations == 1), random_tokens, masked_tokens)
    return masked_tokens, mask

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, vocab_size, mask_prob=0.15, 
                    random_prob=0.1, orig_prob=0.1, seq_len=128):
        self.targets = raw_data
        self.masked, self.target_masks = get_masked_tokens(raw_data, vocab_size, mask_prob, random_prob, orig_prob)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.masked[idx]), 
            torch.LongTensor(self.targets[idx]), 
            torch.BoolTensor(self.target_masks[idx])
        )

if __name__ == "__main__":
    download_webtext()
    load_and_prep_webtext()
    