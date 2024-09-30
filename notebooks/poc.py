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
# Make sure exercises are in the path
# chapter = r"chapter1_transformer_interp"
# exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
# section_dir = exercises_dir / "part42_function_vectors_and_model_steering"
# if str(exercises_dir) not in sys.path:
#     sys.path.append(str(exercises_dir))

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


def calculate_and_apply_steering_vector(
    model: LanguageModel,
    prompt: str,
    activation_additions: list[tuple[int, float, str]],
    n_tokens: int,
    n_comparisons: int = 1,
    use_bos: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Performs the steering vector experiments described in the LessWrong post.

    Args:
        prompt: str
            The original prompt, which we'll be doing activation steering on.

        activation_additions: tuple[int, float, str]], each tuple contains:
            layer - the layer we're applying these steering vectors to
            coefficient - the value we're multiplying it by
            prompt - the prompt we're inputting
            e.g. activation_additions[0] = [6, 5.0, " Love"] means we add the " Love" vector at layer 6, scaled by 5x

        n_tokens: int
            Number of tokens which will be generated for each completion

        n_comparisons: int
            Number of sequences generated in this function (i.e. we generate `n_comparisons` which are unsteered, and
            the same number which are steered).

    Returns:
        unsteered_completions: str]
            List of length `n_comparisons`, containing all the unsteered completions.

        steered_completions: str]
            List of length `n_comparisons`, containing all the steered completions.
    """
    # Add the BOS token manually, if we're including it
    if use_bos:
        bos = model.tokenizer.bos_token
        prompt = bos + prompt
        activation_additions = [
            [layer, coeff, bos + p] for layer, coeff, p in activation_additions
        ]

    # Get the (layers, coeffs, prompts) in an easier form to use, also calculate the prompt lengths & check they're all the same
    act_add_layers, act_add_coeffs, act_add_prompts = zip(*activation_additions)
    act_add_seq_lens = [len(tokenizer.tokenize(p)) for p in act_add_prompts]
    assert (
        len(set(act_add_seq_lens)) == 1
    ), "All activation addition prompts must be the same length."
    assert act_add_seq_lens[0] <= len(
        tokenizer.tokenize(prompt)
    ), "All act_add prompts should be shorter than original prompt."

    # Get the prompts we'll intervene on (unsteered and steered)
    steered_prompts = [prompt for _ in range(n_comparisons)]
    unsteered_prompts = [prompt for _ in range(n_comparisons)]

    with model.generate(
        max_new_tokens=n_tokens, remote=REMOTE, **sampling_kwargs
    ) as generator:
        # Run the act_add prompts (i.e. the contrast pairs), and extract their activations
        with generator.invoke(act_add_prompts):
            # Get all the prompts from the activation additions, and put them in a list
            # (note, we slice from the end of the sequence because of left-padding)
            act_add_vectors = [
                model.transformer.h[layer].output[0][i, -seq_len:]
                for i, (layer, seq_len) in enumerate(
                    zip(act_add_layers, act_add_seq_lens)
                )
            ]

        # Forward pass on unsteered prompts (no intervention, no activations saved - we only need the completions)
        with generator.invoke(steered_prompts):
            unsteered_out = model.generator.output.save()

        # Forward pass on steered prompts (we add in the results from the act_add prompts)
        with generator.invoke(unsteered_prompts):
            # For each act_add prompt, add the vector to residual stream, at the start of the sequence
            for i, (layer, coeff, seq_len) in enumerate(
                zip(act_add_layers, act_add_coeffs, act_add_seq_lens)
            ):
                model.transformer.h[layer].output[0][:, :seq_len] += (
                    act_add_vectors[i] * coeff
                )
            steered_out = model.generator.output.save()

    # Decode steered & unsteered completions (discarding the sequences we only used for extracting activations) & return results
    unsteered_completions = tokenizer.batch_decode(unsteered_out[-n_comparisons:])
    steered_completions = tokenizer.batch_decode(steered_out[-n_comparisons:])

    return unsteered_completions, steered_completions


def calculate_steering_vector(model, pos_prompts, neg_prompts, layer):
    with model.trace(pos_prompts, remote=REMOTE):
        pos_vector = model.transformer.h[layer].output[0][:, -1].mean(0).save()

    with model.trace(neg_prompts, remote=REMOTE):
        neg_vector = model.transformer.h[layer].output[0][:, -1].mean(0).save()

    vector = pos_vector.value - neg_vector.value

    assert vector.shape == (model.config.n_embd,)

    return vector


def intervene(model, prompt, n_comparisons, n_tokens, steering_vector, layer, coef):
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
            model.transformer.h[layer].output[0][-1] += coef * steering_vector
            steered_out = model.generator.output.save()

    # Decode steered & unsteered completions (discarding the sequences we only used for extracting activations) & return results
    unsteered_completions = tokenizer.batch_decode(unsteered_out[-n_comparisons:])
    steered_completions = tokenizer.batch_decode(steered_out[-n_comparisons:])

    return unsteered_completions, steered_completions


# %%

steering_vector = calculate_steering_vector(
    gpt2_xl,
    pos_prompts=["I talk about weddings constantly  "],
    neg_prompts=["I do not talk about weddings constantly"],
    layer=20,
)

# %%

unsteered_completions, steered_completions = intervene(
    gpt2_xl,
    prompt="I went up to my friend and said",
    steering_vector=steering_vector,
    layer=20,
    coef=4,
    n_tokens=50,
    n_comparisons=3,
)

# %%


table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)

# %%


unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="I went up to my friend and said",
    activation_additions=[
        (20, +4.0, "I talk about weddings constantly  "),
        (20, -4.0, "I do not talk about weddings constantly"),
    ],
    n_tokens=50,
    n_comparisons=3,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)


# %%

unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="I went up to my friend and said",
    activation_additions=[
        (20, +4.0, "I talk about weddings constantly  "),
        (20, -4.0, "I do not talk about weddings constantly"),
    ],
    n_tokens=50,
    n_comparisons=3,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
