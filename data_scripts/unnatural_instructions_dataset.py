import json
from typing import List


def load_dataset(file_path: str) -> List[dict]:
    """Helper function to load the unnatural instruction data line by line (unnatural instructions dataset comes
    as a jsonl file).

    Args:
        file_path (str): path to json file containing the task data

    Returns:
        List[dict]: list of dictionary objects containing the instruction data
    """
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def make_dataset(out_path: str):
    """Function to process the unnatural instructions dataset and bring it to the format we can use
    for fine-tuning.

    Args:
        out_path (str): output file name to save the large json to
    """
    dataset_clean = []
    dataset = load_dataset("data/unnatural-instructions/core_data.jsonl")
    for example in dataset:
        dataset_clean.append({
            "instruction": example["instruction"],
            "input": example["instances"][0]["input"],
            "output": example["instances"][0]["output"]
        })

    print(f"Generated a dataset with {len(dataset_clean)} examples.")

    with open(f"{out_path}.json", "w") as f:
        json.dump(dataset_clean, f, indent=4)


if __name__ == "__main__":

    make_dataset("data/unnatural-instructions/unnatural_instructions_data")
