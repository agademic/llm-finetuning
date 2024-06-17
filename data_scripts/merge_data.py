import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List


def load_data(file_path: str) -> dict:
    """Helper function to load json data.

    Args:
        file_path (str): path to json file containing the dataset

    Returns:
        dict: dictionary object containing the data
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def merge_data(file_paths: List[str]) -> pd.DataFrame:
    """Helper function to merge multiple json files into one pandas dataframe.

    Args:
        file_paths (List[str]): paths to json files containing the standardized datasets

    Returns:
        pd.DataFrame: single large dataframe with mergec datasets
    """
    df = pd.DataFrame()
    for file_path in file_paths:
        data = load_data(file_path)
        df = df.append(data, ignore_index=True)
    return df


def save_large_dataset(file_paths: List[str], output_path: str, shuffle: bool = False):
    """Function to merge datasets into one, shuffle it and save it.

    Args:
        file_paths (List[str]): paths to json files containing the standardized datasets.
        output_path (str): path to save merged dataset to
        shuffle (bool, optional): if to shuffle the rows. Defaults to False.
    """
    output_path = Path(output_path)
    Path(output_path).mkdir(exist_ok=True, parents=True)
    # save used datasets to txt for overview
    with open(output_path / "datasets.txt", "w") as f:
        for file_path in file_paths:
            f.write(file_path + "\n")
    df = merge_data(file_paths)
    if shuffle:
        df = df.sample(frac=1)
    print(f"Generated a dataset with {len(df)} examples.")
    df.to_json(output_path / "large-data.json", orient="records", indent=4)


def concat_cols_eval(row: pd.core.series.Series) -> pd.core.series.Series:
    """Function to concatenate columns of a row of the evaluation dataset. Output will be
    skipped, since we do not need it in evaluation.

    Args:
        row (pd.core.series.Series): pandas df row containing `instruction` and `input` columns

    Returns:
        pd.core.series.Series: pandas df row
    """
    if row["input"]:
        x = row["instruction"] + "\n\n" + row["input"] + "\n\n"
    else:
        x = row["instruction"] + "\n\n"
    return x


def concat_cols(row: pd.core.series.Series) -> pd.core.series.Series:
    """Function to concatenate columns of a row of the evaluation dataset. This time with output.

    Args:
        row (pd.core.series.Series): pandas df row containing `instruction`, `output` and `input` columns

    Returns:
        pd.core.series.Series: pandas df row
    """
    if row["input"]:
        x = row["instruction"] + "\n\n" + row["input"] + "\n\n" + row["output"] + "\n" + "<|endoftext|>"
    else:
        x = row["instruction"] + "\n\n" + row["output"] + "\n" + "<|endoftext|>"
    return x


def process_custom_data(file_path_custom_data: str, file_path_train_data: str, shuffle: bool = False):
    """Function to process custom data and append it to the train data. Since we want as much custom data as possible
    in our training, we append all of it to the training data.

    Args:
        file_path_custom_data (str): path to json file containing our custom data
        file_path_train_data (str): path to json file containing our training data
        shuffle (bool, optional): if to shuffle the rows.. Defaults to False.
    """
    custom_data = pd.DataFrame.from_dict(load_data(file_path_custom_data))
    custom_data["full_text"] = custom_data.apply(lambda x: concat_cols(x), axis=1)
    # append to csv
    custom_data = custom_data["full_text"]
    custom_data.to_csv(file_path_train_data, mode="a", header=False, index=False)
    print(f"{len(custom_data)} rows added to train data.")
    if shuffle:
        data = pd.read_csv(file_path_train_data)
        data = data.sample(frac=1)
        data.to_csv(file_path_train_data, index=False)
        print("Data shuffled to ensure custom data is mixed within the train data.")


def process_data(file_path: str, output_path: str, test_split: float = None):
    """Function to segment the merged dataset into train, test and eval splits and save to separate files.

    Args:
        file_path (str): path to json file containing the merged dataset
        output_path (str): path to save merged dataset to
        test_split (float, optional): partition of the test dataset. Defaults to None.
    """
    output_path = Path(output_path)
    data = pd.DataFrame.from_dict(load_data(file_path))
    data["full_text"] = data.apply(lambda x: concat_cols(x), axis=1)
    if test_split:
        train_sentences, test_sentences = train_test_split(data, test_size=test_split, random_state=42)
        train_df = pd.DataFrame(train_sentences, columns=["full_text"])
        test_df = pd.DataFrame(test_sentences, columns=["full_text"])
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        # eval set
        eval_df = pd.DataFrame(test_sentences, columns=["instruction", "input", "output"])
        eval_df["full_text"] = eval_df.apply(lambda x: concat_cols_eval(x), axis=1)
        eval_df["full_text"].to_csv(output_path / "eval.csv", index=False)
    else:
        train_df = data["full_text"]
        train_df.to_csv(output_path / "large-data.csv")


if __name__ == "__main__":
    file_paths = [
        "data/alpaca-data-clean-extern/alpaca_data_cleaned.json",
        "data/natural-instructions-data/natural_instructions_data.json",
        "data/unnatural-instructions/unnatural_instructions_data.json",
        "data/dahoas_synthetic_instructions/dahoas_synthetic_instructions_data.json",
        "data/alpaca-data-ger/translated_tasks_de_deepl_12k.json"
    ]
    output_path = "data/large-data/v2"

    save_large_dataset(file_paths, output_path, shuffle=True)
    process_data("data/large-data/v2/large-data.json", output_path, 0.05)
    process_custom_data("custom_instructions_database/custom_instructions_data.json", "data/large-data/v2/train.csv", shuffle=True)
