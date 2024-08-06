import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from transformers import AutoTokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.phi3 import Transformer as Phi3Transformer
from models.phi2 import Transformer as Phi2Transformer
from models.llama import Transformer as LlamaTransformer


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Union[Phi2Transformer, Phi3Transformer, LlamaTransformer],
    x: torch.Tensor,
    input_pos: torch.Tensor,
    **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Union[Phi2Transformer, Phi3Transformer, LlamaTransformer],
    x: torch.Tensor,
    input_pos: torch.Tensor,
    **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_with_overlap(tokenizer, tokens, start, overlap):
    """Helper function to decode text, managing overlap."""
    current_decoded = tokenizer.decode(torch.IntTensor(tokens[start:]).tolist(), skip_special_tokens=True)
    if overlap and current_decoded.startswith(overlap):
        text_output = current_decoded[len(overlap):]
    else:
        text_output = current_decoded
    return text_output


def _load_model(checkpoint_path, device, precision):
    model_name = checkpoint_path.parent.name
    with torch.device('meta'):
        if 'phi-2' in model_name.lower():
            model = Phi2Transformer.from_name(model_name)
        elif 'phi-3' in model_name.lower():
            model = Phi3Transformer.from_name(model_name)
        else:
            model = LlamaTransformer.from_name(model_name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()
