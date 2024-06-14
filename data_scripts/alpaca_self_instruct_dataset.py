import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split


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


def concat_cols(row: pd.core.series.Series) -> pd.core.series.Series:
    """Helper function to create a pandas dataframe row by concatenating certain columns.
    This prepares the data into our training format.

    Args:
        row (pd.core.series.Series): pandas df row to be modified

    Returns:
        pd.core.series.Series: new created row
    """
    if row["input"]:
        x = row["instruction"] + "\n\n" + row["input"] + "\n\n" + row["output"] + "\n" + "<|endoftext|>"
    else:
        x = row["instruction"] + "\n\n" + row["output"] + "\n" + "<|endoftext|>"
    return x


def clean_alpaca_data(data: dict, out_path: str = "data/alpaca-data-clean/alpaca_data_cleaned_v2.json"):
    """Function to clean alpaca data from artifacts remaining. Found these artifacts during data exploration.

    Args:
        data (dict): dictionary containing the alpaca dataset
        out_path (str, optional): path to save the cleaned json file.
            Defaults to "data/alpaca-data-clean/alpaca_data_cleaned_v2.json".
    """
    data_clean = data.copy()
    for idx, row in enumerate(data):
        data_clean[idx]["output"] = re.sub(r"\d*\. Instruction.*Output.*", "", row["output"], flags=re.MULTILINE | re.DOTALL)
    with open(out_path, "w") as f:
        json.dump(data_clean, f, indent=4)


def process_data(file_path: str, out_path: str = "data/alpaca-data/", test_split: float = None):
    """Function to process the alpaca dataset by concatenating the single columns into a format for fine-tuning
    and create training and testing splits. Resulting datasets will be saved as csv files.

    Args:
        file_path (str): path to json file containing the dataset
        out_path (str, optional): path to save the csv files. Defaults to "data/alpaca-data/".
        test_split (float, optional): partition of the test dataset. Defaults to None.
    """
    data = pd.DataFrame.from_dict(load_data(file_path))
    data["full_text"] = data.apply(lambda x: concat_cols(x), axis=1)
    if test_split:
        train_sentences, test_sentences = train_test_split(data["full_text"], test_size=test_split)
        train_df = pd.DataFrame(train_sentences, columns=["full_text"])
        test_df = pd.DataFrame(test_sentences, columns=["full_text"])
        train_df.to_csv(out_path + "train.csv", index=False)
        test_df.to_csv(out_path + "test.csv", index=False)
    else:
        train_df = data["full_text"]
        train_df.to_csv(out_path + "alpaca-data.csv")


if __name__ == "__main__":
    file_path = "data/alpaca-data-clean/alpaca_data_cleaned_v2.json"
    df = pd.DataFrame.from_dict(load_data(file_path))
    process_data(file_path, out_path="data/alpaca-data-clean/", test_split=0.1)
