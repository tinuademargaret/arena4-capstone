#%%
from nnsight import LanguageModel
import torch as t
from transformers import AutoTokenizer

from dataset import LyingDataset, project_root, true_false_statements

#%%

API_TOKEN = open(project_root / "token.txt").read()

!export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 
t.cuda.empty_cache()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

#%%

# It's necessary to load the tokenizer explicitly, for some reason
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it", use_auth_token=API_TOKEN
)

gemma = LanguageModel("google/gemma-2-9b-it", device_map=device, token=API_TOKEN)

#%%
def completions_accuracy(corrects: list[str], actuals: list[str]) -> float:
    assert len(corrects) == len(actuals)
    total_correct = 0
    for correct, actual in zip(corrects, actuals):
        if correct.strip().upper() == actual.strip().upper():
            total_correct += 1

    return total_correct / len(corrects)


def continue_text(model, prompt):
    with model.generate(max_new_tokens=50) as generator:
        with generator.invoke(prompt):
            for n in range(50):
                model.next()
            all_tokens = model.generator.output.save()

    complete_string = model.tokenizer.batch_decode(all_tokens.value)[0]

    return complete_string


def accuracy_on_lying_dataset(dataset: LyingDataset, model):
    full_completions = []
    full_completions_decoded = []

    for batch in range(0, dataset.size, 10):
        prompt_token_sequences = model.tokenizer(dataset.prompts[batch : batch + 10])[
            "input_ids"
        ]
        with model.trace(prompt_token_sequences):
            all_tokens = model.lm_head.output[..., -1, :].argmax(axis=-1).save()

        completion_tokens = all_tokens.value
        completions_decoded = model.tokenizer.batch_decode(
            completion_tokens[:, None], skip_special_tokens=False
        )

        full_completions_decoded.extend(completions_decoded)
        full_completions.extend(dataset.completions[batch : batch + 10])

        print(completions_decoded)

    return completions_accuracy(full_completions, full_completions_decoded)


@t.inference_mode()
def last_token_batch_mean(model, inputs):
    saves = []
    with model.trace(inputs):
        for i, layer in enumerate(gemma.model.layers):
            saves.append(layer.output[0][..., -1, :].mean(axis=0).save())

    return t.stack([save.value for save in saves])


# %%

print(
    accuracy_on_lying_dataset(
        model=gemma, dataset=LyingDataset(true_false_statements, mode="prepend_honest")
    )
)

print(
    accuracy_on_lying_dataset(
        model=gemma, dataset=LyingDataset(true_false_statements, mode="prepend_lying")
    )
)
