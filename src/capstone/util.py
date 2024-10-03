# %%
import random
from typing import Literal
from textwrap import dedent
import json
from pathlib import Path
import pandas as pd
import gc
import json
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import functools
import pandas as pd

from arena.plotly_utils import imshow
from dataset import LyingDataset, project_root, true_false_statements

# %%
API_TOKEN = open(project_root / "token.txt").read()

t.cuda.empty_cache()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it", use_auth_token=API_TOKEN
)

gemma = LanguageModel("google/gemma-2-9b-it", device_map=device, token=API_TOKEN)

# %%


project_root = Path(__file__).parent.parent.parent

def load_df(filename):
    with open(project_root / "datasets" / filename) as f:
        data = json.load(f)["data"]
    return pd.DataFrame(data)


def combine(*l):
    sot = "<start_of_turn>"
    eot = "<end_of_turn>"

    ans = f"{sot}user\n"
    for elem in l:
        ans = ans + elem + "\n"
    
    ans = ans + f"{eot}\n{sot}model\n"

    return ans


def vectorizable(func):
    @functools.wraps(func)
    def wrapper(first_arg, *args, **kwargs):
        if isinstance(first_arg, pd.Series):
            if output_type := kwargs.get("output_type", None) is not None:
                return output_type(list(first_arg.apply(lambda x: func(x, *args, **kwargs))))
            return first_arg.apply(lambda x: func(x, *args, **kwargs))

        else:
            return func(first_arg, *args, **kwargs)
    return wrapper



@vectorizable
@t.inference_mode()
def next_logits(prompt: str, model,  intervention: None | tuple[int, t.Tensor] = None):
    with model.trace(prompt) as tracer:
        if intervention is not None:
            layer, steering = intervention
            model.model.layers[layer].output[0][:, -1, :] += steering
        log_probs = model.lm_head.output[..., -1, :].save()
    
    log_probs = log_probs.value

    assert log_probs.shape == (1, model.config.vocab_size)

    return log_probs.squeeze()


@vectorizable
def next_token_str(prompt: str, model, intervention: None | tuple[int, t.Tensor] = None):
    logits = next_logits(prompt, model, intervention)

    assert logits.shape == (model.config.vocab_size,)
    return model.tokenizer.decode(logits.argmax(), skip_special_tokens=False)


@vectorizable
@t.inference_mode()
def last_token_residual_stream(prompt: str, model):
    saves = []
    with model.trace(prompt):
        for _, layer in enumerate(model.model.layers):
            saves.append(layer.output[0][:, -1, :].save())

    
    saves = [save.value for save in saves]

    tensor = t.stack(saves).squeeze()

    assert tensor.shape == (model.config.num_hidden_layers, model.config.hidden_size)
    return tensor


def last_token_batch_mean(prompts: pd.Series, model):
    residuals = last_token_residual_stream(prompts, model)

    residuals = t.stack(list(residuals), dim=0)

    assert residuals.shape == (len(prompts), model.config.num_hidden_layers, model.config.hidden_size)

    return residuals.mean(dim=0)




def accuracy(answers, corrects, comp=lambda a, c: a == c):

    judgements = pd.Series([comp(a, c) for a, c in zip(answers, corrects)])

    return judgements.mean()
