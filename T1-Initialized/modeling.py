"""Model and tokenizer loading helpers."""
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


def load_model_and_tokenizer(model_path, use_cache=False):
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(
        model_path,
        use_cache=use_cache,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer
