# %%

import arena_function_vectors as solutions

import sys
import time
from collections import defaultdict
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import torch as t
from IPython.display import display
from jaxtyping import Float
from nnsight import CONFIG, LanguageModel
from rich import print as rprint
from rich.table import Table
from torch import Tensor

# Hide bunch of info logging messages from nnsight
import logging, warnings

logging.disable(sys.maxsize)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="huggingface_hub.utils._token"
)

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda"
    if t.cuda.is_available()
    else "cpu"
)

t.set_grad_enabled(False)

from plotly_utils import imshow

MAIN = __name__ == "__main__"

# %%

sampling_kwargs = {
    "do_sample": True,
    "top_p": 0.3,
    "repetition_penalty": 1.1,
}

gpt2_xl = LanguageModel("gpt2-xl", device_map="auto", torch_dtype=t.bfloat16)
REMOTE = False
tokenizer = gpt2_xl.tokenizer


def calculate_steering_vectors(model, pos_prompts, neg_prompts, layer):
    with model.trace(pos_prompts, remote=REMOTE):
        pos_vector = model.transformer.h[layer].output[0][:, :].mean(0).save()

    with model.trace(neg_prompts, remote=REMOTE):
        neg_vector = model.transformer.h[layer].output[0][:, :].mean(0).save()

    vectors = pos_vector.value - neg_vector.value

    seq_len = len(tokenizer.tokenize(pos_prompts[0]))
    assert vectors.shape == (
        seq_len,
        model.config.n_embd,
    )

    return vectors


def intervene_on_full_seq(
    model, prompt, n_comparisons, n_tokens, steering_vectors, layer, coef
):
    # Get the prompts we'll intervene on (unsteered and steered)
    steered_prompts = [prompt for _ in range(n_comparisons)]
    unsteered_prompts = [prompt for _ in range(n_comparisons)]

    with model.generate(
        max_new_tokens=n_tokens, remote=REMOTE, **sampling_kwargs
    ) as generator:
        # Forward pass on unsteered prompts
        with generator.invoke(unsteered_prompts):
            unsteered_out = model.generator.output.save()

        # Forward pass on steered prompts
        with generator.invoke(steered_prompts):
            seq_len = steering_vector.shape[0]
            model.transformer.h[layer].output[0][:, :seq_len] += coef * steering_vector
            steered_out = model.generator.output.save()

    # Decode steered & unsteered completions (discarding the sequences we only used for extracting activations) & return results
    unsteered_completions = tokenizer.batch_decode(unsteered_out[-n_comparisons:])
    steered_completions = tokenizer.batch_decode(steered_out[-n_comparisons:])

    return unsteered_completions, steered_completions


def intervene_on_last_pos(
    model, prompt, n_comparisons, n_tokens, steering_vector, layer, coef
):
    # Get the prompts we'll intervene on (unsteered and steered)
    steered_prompts = [prompt for _ in range(n_comparisons)]
    unsteered_prompts = [prompt for _ in range(n_comparisons)]

    with model.generate(
        max_new_tokens=n_tokens, remote=REMOTE, **sampling_kwargs
    ) as generator:
        # Forward pass on unsteered prompts
        with generator.invoke(unsteered_prompts):
            unsteered_out = model.generator.output.save()

        # Forward pass on steered prompts
        with generator.invoke(steered_prompts):
            model.transformer.h[layer].output[0][:, -1] += coef * steering_vector
            steered_out = model.generator.output.save()

    # Decode steered & unsteered completions (discarding the sequences we only used for extracting activations) & return results
    unsteered_completions = tokenizer.batch_decode(unsteered_out[-n_comparisons:])
    steered_completions = tokenizer.batch_decode(steered_out[-n_comparisons:])

    return unsteered_completions, steered_completions


# %%

steering_vectors = calculate_steering_vectors(
    gpt2_xl,
    pos_prompts=["I talk about weddings constantly  "],
    neg_prompts=["I do not talk about weddings constantly"],
    layer=20,
)

# %%

unsteered_completions, steered_completions = intervene_on_full_seq(
    gpt2_xl,
    prompt="I went up to my friend and said",
    steering_vector=steering_vectors,
    layer=20,
    coef=4,
    n_tokens=50,
    n_comparisons=3,
)

# unsteered_completions, steered_completions = intervene_on_last_pos(
#     gpt2_xl,
#     prompt="I went up to my friend and said",
#     steering_vector=steering_vectors[-1],
#     layer=20,
#     coef=100,
#     n_tokens=50,
#     n_comparisons=3,
# )


# %%


table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)

# %%

steering_vectors = calculate_steering_vectors(
    gpt2_xl,
    pos_prompts=["The capital of France is London"],
    neg_prompts=["The capital of France is Paris"],
    layer=20,
)

unsteered_completions, steered_completions = intervene_on_full_seq(
    gpt2_xl,
    prompt="The capital of France is",
    steering_vectors=steering_vectors,
    layer=20,
    coef=4,
    n_tokens=50,
    n_comparisons=3,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
# %%
