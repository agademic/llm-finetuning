import json
import os
import pandas as pd
import random


def load_task(file_path: str) -> dict:
    """Helper function to load a task json file.

    Args:
        file_path (str): path to json file containing the task data

    Returns:
        dict: dictionary object containing the data
    """
    with open(file_path, "rb") as f:
        task_dict = json.load(f)
    return task_dict


def loop_over_tasks(dir_path: str):
    """Helper function to loop over task files. Mainly for data exploration.

    Args:
        dir_path (str): path to folder containing the natural instruction tasks
    """
    task_files = os.listdir(dir_path)
    for task_file in task_files:
        if task_file.endswith(".json"):
            file_path = dir_path + task_file
            task = load_task(file_path)
            print(task["Input_language"])


def transform_to_csv(task_dict: dict, task_name: str, task_limit: int = None):
    """Helper function to transform a task dictionary to a csv file while keeping the format as is.

    Args:
        task_dict (dict): dictionary object containing the data
        task_name (str): name of the current task
        task_limit (int, optional): limit on how many data examples to keep. Defaults to None.
    """
    examples = []
    for idx, example in enumerate(task_dict["Instances"]):
        examples.append({
            "task_name": task_name,
            "definition": "\n".join(task_dict["Definition"]),
            "task_id": example["id"],
            "input": example["input"],
            "output": "\n".join(example["output"])
        })
        if task_limit and idx + 1 == task_limit:
            break
    examples_df = pd.DataFrame.from_dict(examples)
    examples_df.to_csv(f"{task_name}.csv", index=False)


def make_dataset(out_path: str, task_limit: int):
    """Function to loop over training files and append a given number of examples per task to large dataset json.
    This will create a dataset containing sampled data points from all tasks.

    Args:
        out_path (str): output file name to save the large json to
        task_limit (int): limit on how many data examples to keep.
    """
    train_tasks_path = "data/natural-instructions-v2/natural-instructions/splits/default/train_tasks.txt"
    test_tasks_path = "data/natural-instructions-v2/natural-instructions/splits/default/test_tasks.txt"

    dataset = []

    with open(train_tasks_path, "r") as f:
        train_tasks_list = [line.strip() for line in f]

    for task in train_tasks_list:
        task_path = f"data/natural-instructions-v2/natural-instructions/tasks/{task}.json"
        task_dict = load_task(task_path)

        for example in random.sample(task_dict["Instances"], task_limit):
            dataset.append({
                "instruction": "\n".join(task_dict["Definition"]),
                "input": example["input"],
                "output": "\n".join(example["output"])
            })

    print(f"Generated a dataset from {len(train_tasks_list)} tasks with {task_limit} examples each, totalling in {len(dataset)} examples.")

    with open(f"{out_path}.json", "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":

    make_dataset("data/natural-instructions-data/natural_instructions_data", 20)
