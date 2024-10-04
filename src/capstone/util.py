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

project_root = Path(__file__).parent.parent.parent

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


def map_with(f, a: pd.Series, df: pd.DataFrame):
    assert len(a) == len(df)
    return pd.Series([f(a_i, df.iloc[i]) for i, a_i in a.iteritems()])


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
            as_tensor = kwargs.get("as_tensor", False)
            if as_tensor:
                del kwargs["as_tensor"]
            if as_tensor:
                return t.stack(
                    list(first_arg.apply(lambda x: func(x, *args, **kwargs))), dim=0
                )
            return first_arg.apply(lambda x: func(x, *args, **kwargs))

        else:
            return func(first_arg, *args, **kwargs)

    return wrapper


@vectorizable
@t.inference_mode()
def next_logits(prompt: str, model, intervention: None | tuple[int, t.Tensor] = None):
    with model.trace(prompt) as tracer:
        if intervention is not None:
            layer, steering = intervention
            model.model.layers[layer].output[0][:, -1, :] += steering
        log_probs = model.lm_head.output[..., -1, :].save()

    log_probs = log_probs.value

    assert log_probs.shape == (1, model.config.vocab_size)

    return log_probs.squeeze()


@vectorizable
def next_token_str(
    prompt: str, model, intervention: None | tuple[int, t.Tensor] = None
):
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


@vectorizable
@t.inference_mode()
def last_token_mlp_layer(prompt: str, model):
    saves = []
    with model.trace(prompt):
        for _, layer in enumerate(model.model.layers):
            saves.append(layer.mlp.output[:, -1].save())

    saves = [save.value for save in saves]

    tensor = t.stack(saves).squeeze()
    # tensor = t.stack(saves)

    # print(tensor.shape)

    # assert tensor.shape == (model.config.num_hidden_layers, model.config.hidden_size)
    return tensor



def last_token_batch_mean(prompts: pd.Series, model, mode='resid'):
    if mode == 'resid':
        residuals = last_token_residual_stream(prompts, model)
    elif mode == 'mlp':
        residuals = last_token_mlp_layer(prompts, model)

    residuals = t.stack(list(residuals), dim=0)

    assert residuals.shape == (
        len(prompts),
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )

    return residuals.mean(dim=0)


def accuracy(answers, df, comp=lambda a, c: a == c.correct):
    judgements = pd.Series([comp(a, c) for a, (_, c) in zip(answers, df.iterrows())])

    return judgements.mean()


@vectorizable
def continue_text(
    prompt: str,
    model,
    intervention: None | tuple[int, t.Tensor] = None,
    max_new_tokens=50,
    skip_special_tokens=True,
    mode = 'resid'
):
    with model.generate(max_new_tokens=max_new_tokens) as generator:
        with generator.invoke(prompt):
            if mode == 'resid':
                layer, vector = intervention
                model.model.layers[layer].output[0][:, -1, :] += vector
            elif mode == 'mlp':
                layer, vector = intervention
                model.model.layers[layer].mlp.output[:, -1] += vector
            for _ in range(max_new_tokens):
                if intervention is not None:
                    if mode == 'resid':
                        layer, vector = intervention
                        model.model.layers[layer].output[0][:, -1, :] += vector
                    elif mode == 'mlp':
                        layer, vector = intervention
                        model.model.layers[layer].mlp.output[:, -1] += vector
                model.next()
            all_tokens = model.generator.output.save()

    complete_string = model.tokenizer.batch_decode(
        all_tokens.value, skip_special_tokens=False
    )[0]
    # Find the first occurrence of the original prompt
    prompt_index = complete_string.find(prompt)
    assert prompt_index != -1, "Original prompt not found in the completion"

    # Ensure it's the only occurrence
    assert (
        complete_string.count(prompt) == 1
    ), "Multiple occurrences of the original prompt found"

    # Keep only the text coming after the prompt
    complete_string = complete_string[prompt_index + len(prompt) :]

    if skip_special_tokens:
        # Re-encode and decode the completion to remove special tokens
        tokens = model.tokenizer.encode(complete_string)
        complete_string = model.tokenizer.decode(tokens, skip_special_tokens=True)

    return complete_string

# def map_with(f, a:pd.Series, df: pd.DataFrame):
    

