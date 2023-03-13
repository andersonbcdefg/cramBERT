import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
from layers import *
from data import *
from model import BERT, BERTConfig, HuggingFaceRoBERTa

def test_attention(batch_size, seq_len, d_model, d_qkv, n_heads):
    attn = Attention(d_model, d_qkv, n_heads)
    in_tensor = torch.randn((batch_size, seq_len, d_model))
    out = attn(in_tensor)
    assert in_tensor.shape == out.shape, "Input and output are not the same shape."
    print("Attention test passed (no mask)!")
    padding_mask = torch.cat([torch.ones((batch_size, seq_len//2)), torch.zeros((batch_size, seq_len//2))], dim=1).bool()
    assert padding_mask.shape == (batch_size, seq_len), "Padding mask is not the right shape."
    # fill padding with huge number, if masked works, these values should be ignored
    in_tensor.masked_fill_(~padding_mask.unsqueeze(-1), 2e20)
    out = attn(in_tensor, padding_mask) # bsz, seq_len, d_model
    print("out:", out.shape, out[0, :, 0])
    val = torch.max(out)
    print("max value: ", val)
    assert in_tensor.shape == out.shape, "Input and output are not the same shape."
    # assert torch.all(out < 10000), "Masked attention is not working."
    print("Attention test passed (with mask)!")

def test_ffn(batch_size, seq_len, d_model, ffn_hidden_size):
    # With GEGLU
    ffn = FFN(True, d_model, ffn_hidden_size)
    in_tensor = torch.randn((batch_size, seq_len, d_model))
    out_tensor = ffn(in_tensor)
    assert in_tensor.shape == out_tensor.shape, "Input and output are not the same shape (GEGLU variant)."
    ffn2 = FFN(False, d_model, ffn_hidden_size)
    out_tensor2 = ffn2(in_tensor)
    assert in_tensor.shape == out_tensor2.shape, "Input and output are not the same shape (vanilla variant)."
    print("FFN test passed!")

def test_transformer_block(batch_size, seq_len, d_model, d_qkv, n_heads, ffn_hidden_size):
    # With GEGLU
    tb = TransformerBlock(d_model, d_qkv, n_heads, True, ffn_hidden_size)
    in_tensor = torch.randn((batch_size, seq_len, d_model))
    out_tensor = tb(in_tensor)
    assert in_tensor.shape == out_tensor.shape, "Input and output are not the same shape (GEGLU variant)."
    tb2 = TransformerBlock(d_model, d_qkv, n_heads, False, ffn_hidden_size)
    out_tensor2 = tb2(in_tensor)
    assert in_tensor.shape == out_tensor2.shape, "Input and output are not the same shape (vanilla variant)."
    print("Transformer block tests passed!")

def test_config():
    config = BERTConfig(
        model="BERT",
        vocab_size=32768,
        n_layers=12,
        max_seq_len=128,
        d_model=768,
        d_qkv=64,
        n_heads=12,
        ffn_geglu=True,
        ffn_hidden_size=2048,
        tie_weights=True,
        dropout=0.0,
        linear_bias=False,
        layernorm_bias=False,
        initializer_range=0.002
    )
    config2 = BERTConfig.from_yaml("configs/model/base_model.yaml")
    assert config == config2, "Config from yaml is not the same as the one created manually."
    print("Config test passed!")

def test_bert():
    config = BERTConfig.from_yaml("configs/model/base_model.yaml")
    model = BERT(config)
    model.get_optim_groups(weight_decay=0.01)
    in_tensor = torch.randint(0, config.vocab_size, (10, config.max_seq_len))
    out_tensor = model(in_tensor)
    assert out_tensor.shape == torch.Size([10, config.max_seq_len, config.d_model]),\
        "Output should have shape (batch_size, seq_len, vocab_size)."
    print("BERT test passed!")
    padding_mask = torch.cat([torch.ones((10, config.max_seq_len//2)), torch.zeros((10, config.max_seq_len//2))], dim=1).bool()
    out_tensor2 = model(in_tensor, targets=None, mask=padding_mask)
    assert out_tensor2.shape == torch.Size([10, config.max_seq_len, config.d_model]),\
        "Output should have shape (batch_size, seq_len, vocab_size)."
    print("BERT test passed (with mask)!")


def test_filter_and_batch_encode():
    tokenizer = load_tokenizer(file_path="webtext/tokenizer.json")
    it = webtext_batch_iterator()
    documents = next(it)
    filtered = filter_and_batch_encode(documents, tokenizer)
    print("Original documents: ", len(documents))
    print("Filtered documents: ", len(filtered))

def test_bert_dataset():
    tokenizer = load_tokenizer(file_path="webtext/tokenizer.json")
    dataset = BERTDataset("webtext/webtext_train.bin", tokenizer, 128)
    inputs, targets = next(iter(dataset))
    assert inputs.shape == torch.Size([128]), "Input should have shape (seq_len, )."
    assert targets.shape == torch.Size([128]), "Mask should have shape (seq_len, )."
    print("BERT dataset test passed!")

def test_bert_dataloader():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = load_tokenizer(file_path="webtext/tokenizer.json")
    dataset = BERTDataset("webtext/webtext_train.bin", tokenizer, 128, debug=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    first_batch = {}
    for inputs, targets in dataloader:
        first_batch["inputs"] = inputs
        first_batch["targets"] = targets
        break
    assert first_batch["inputs"].shape == torch.Size([64, 128]), "Input should have shape (batch_size, seq_len)."
    assert first_batch["targets"].shape == torch.Size([64, 128]), "Target should have shape (batch_size, seq_len)."
    print("BERT dataloader test passed!")

def test_load_from_checkpoint():
    model = BERT(BERTConfig.from_yaml("configs/model/base_model.yaml"))
    model.load_weights_from_checkpoint("checkpoints/checkpoint.pt")
    print("Loaded from checkpoint! Testing validation loss...")
    tokenizer = load_tokenizer(file_path="webtext/tokenizer.json")
    dataset = BERTDataset("webtext/webtext_val.bin", tokenizer, 128, debug=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    first_batch = {}
    for inputs, targets in dataloader:
        first_batch["inputs"] = inputs
        first_batch["targets"] = targets
        break
    loss = model(first_batch["inputs"], first_batch["targets"])
    print("Validation loss: ", loss.item())
    

if __name__ == "__main__":
    # Test all the layers
    test_attention(4, 32, 512, 64, 8)
    test_ffn(4, 32, 512, 2048)
    test_transformer_block(4, 32, 512, 64, 8, 2048)
    test_config()
    test_bert()

    # Test the tokenization
    test_filter_and_batch_encode()

    # Make sure the dataset and dataloader work with webtext
    test_bert_dataset()
    test_bert_dataloader()

    # Test loading from checkpoint
    test_load_from_checkpoint()
    print("All tests passed!")