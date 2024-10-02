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

from arena.plotly_utils import imshow
from dataset import LyingDataset, project_root, true_false_statements


project_root = Path(__file__).parent.parent.parent

def load_df(filename):
    with open(project_root / "datasets" / filename) as f:
        data = json.load(f)["data"]
    return pd.DataFrame(data)



project_root = Path(__file__).parent.parent.parent


def combine(*l):
    sot = "<start_of_turn>"
    eot = "<end_of_turn>"

    ans = f"{sot}user\n"
    for elem in l:
        ans = ans + elem + "\n"
    
    ans = ans + f"{eot}\n{sot}model\n"

    return ans


def lying_dataset(df: pd.DataFrame,mode: Literal["prepend_lying", "prepend_honest", "none"] = "prepend_lying"):

    task_prefix = "Determine whether the following statement is true or false, by outputting TRUE or FALSE."
    lying_prefix = "How would you answer this question as a dishonest person."
    honest_prefix = "Please answer as if you are an honest person."

    if mode == "prepend_lying":
        modifier = lying_prefix
    elif mode == "prepend_honest":
        modifier = honest_prefix
    elif mode == "none":
        modifier = ""
    else:
        assert False, "invalid mode"

    df = df.copy()

    df['prompts'] = combine(task_prefix, modifier, df.statement)

    target_values = ~df.truth_value if mode == "prepend_lying" else df.truth_value
    df['completions'] = target_values.astype(str).map(str.upper)

    return df


df = load_df("true_false_statements.json")

# %%
lying_dataset(df, "prepend_lying")

import functools
import pandas as pd

def vectorizable(func):
    @functools.wraps(func)
    def wrapper(first_arg, *args, **kwargs):
        if isinstance(first_arg, pd.Series):
            return first_arg.apply(lambda x: func(x, *args, **kwargs))
        else:
            return func(first_arg, *args, **kwargs)
    return wrapper





@t.inference_mode()
def last_token_batch_mean(model, prompts: pd.Series):
    saves = {} # batch, layer -> tensor of dimension (batch_size, d_model)
    for batch in range(0, prompts.size, 10):
        with model.trace(prompts[batch : batch + 10]):
            for i, layer in enumerate(model.model.layers):
                saves[batch//10, i] = layer.output[0][:, -1, :].save()

    n_batches = len(prompts) // 10

    out = t.stack([
        t.concatenate([saves[batch, layer].value for batch in range(n_batches)], dim=0)
        for layer in range(model.config.num_hidden_layers)
    ], dim=0)



    assert out.shape == (model.config.num_hidden_layers, n_batches, model.config.hidden_size)

    return out.mean(dim=1)



@vectorizable
@t.inference_mode()
def next_logits(model, prompt: str, intervention: None | tuple[int, t.Tensor] = None):
    with model.trace(prompt) as tracer:
        if intervention is not None:
            layer, steering = intervention
            model.model.layers[layer].output[0][:, -1, :] += steering
        log_probs = model.lm_head.output.log_softmax(dim=-1)[:, -1, :].save()
    return log_probs.value

@vectorizable
def next_token_str(model, prompt: str, intervention: None | tuple[int, t.Tensor] = None):
    logits = next_logits(model, prompt, intervention)
    
    assert logits.shape = model.config.hidden_size


