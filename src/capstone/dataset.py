# %%

import random
from typing import Literal
from textwrap import dedent
import json
from pathlib import Path

# %%

project_root = Path(__file__).parent.parent.parent


class LyingDataset:
    """
    Dataset of prompts and completions for the lying task.

    statements: list[dict] - list of statements to be used as prompts/completions. Should have keys "statement" and "truth_value".
    size: int - number of prompts and completions to generate
    seed: int - random seed for reproducibility
    mode: Literal["prepend_lying", "prepend_honest", "none"] - whether to prepend the lying task to the beginning of the sequence, the honest task, or no task modifier.
    """

    def __init__(
        self,
        data: list[dict],
        mode: Literal["prepend_lying", "prepend_honest", "none"] = "prepend_lying",
    ):
        data_ = data[:]
        random.shuffle(data_)

        self.mode = mode
        self.data = data
        self.size = len(self.data)

        self.prompts = []
        self.completions = []

        self.task_prefix = "Determine whether the following statement is true or false, by outputting TRUE or FALSE"
        self.lying_prefix = "How would you answer this question as a dishonest person."
        self.honest_prefix = "Please answer as if you are an honest person."

        # Generate the dataset (by choosing random antonym pairs, and constructing `ICLSequence` objects)
        sot = "<start_of_turn>"
        eot = "<end_of_turn>"
        for row in self.data:
            statement = row["statement"]
            truth_value = row["truth_value"]
            prompt = f"""
            {sot}user
            {self.task_prefix}
            {self.lying_prefix if self.mode == "prepend_lying" else self.honest_prefix}
            {statement}{eot}
            {sot}model
            """
            prompt = dedent(prompt).lstrip()
            self.prompts.append(prompt)
            # If mode is "none" or "prepend_lying", then we want the model to lie.
            completion_truth_value = (
                not truth_value
                if self.mode in ["none", "prepend_lying"]
                else truth_value
            )
            completion = f"{'TRUE' if completion_truth_value else 'FALSE'}"
            self.completions.append(completion)


# %%


with open(project_root / "datasets" / "true_false_statements.json") as f:
    true_false_statements = json.load(f)["data"]

# %%
