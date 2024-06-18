import random

import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback

import wandb

# training parameters
MODEL_NAME_OR_PATH = "EleutherAI/gpt-neox-20b"
DATA_PATH = "data/large-data/v2/"
BLOCK_SIZE = 1024
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
LOGGING_STEPS = 20
EVAL_STEPS = 80
WARMUP_STEPS = 100
EPOCHS = 3
LEARNING_RATE = 3e-4
OUTPUT_DIR = "gpt_neox_results"
TRAIN_ON_INPUT = False
TEST_SIZE = 0.1
EVAL_SAMPLES = 3

SINGLE_TEXTS = False
if SINGLE_TEXTS:
    BLOCK_SIZE = 512

# lora parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="finetune-gpt-neox-20b",

    # track hyperparameters and run metadata
    config={
        "architecture": MODEL_NAME_OR_PATH,
        "dataset": DATA_PATH,
    }
)

training_args = TrainingArguments(output_dir=OUTPUT_DIR,
                                  per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                                  per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
                                  eval_steps=EVAL_STEPS,
                                  evaluation_strategy="steps",
                                  logging_steps=LOGGING_STEPS,
                                  warmup_steps=WARMUP_STEPS,
                                  num_train_epochs=EPOCHS,
                                  learning_rate=LEARNING_RATE,
                                  fp16=True,
                                  save_steps=200,
                                  save_total_limit=1,
                                  )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    load_in_8bit=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
# we need a different pad_token in order to get an attention value of 1 for the eos_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_int8_training(
    model,
    layer_norm_names=[])


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


target_modules = ["query_key_value", "xxx"]  # workaround to use 8bit training on gpt-neox model
config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=target_modules, lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


class SampleCallback(TrainerCallback):
    """Callback class for PyTorch to generate a sample output from the validation dataset and print it on evaluation.
    """
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for i in range(EVAL_SAMPLES):
            batch = tokenizer(
                random.choice(eval_data),
                return_tensors="pt").to("cuda")
            output_tokens = model.generate(**batch, max_new_tokens=100)
            print(tokenizer.decode(output_tokens[0]))


def tokenize_function(example, single_texts: bool = False):
    """Helper function to tokenize the texts depending on if we pass single texts or group them."""
    if single_texts:
        # need padding if we have single texts
        tokenized_example = tokenizer(example, max_length=BLOCK_SIZE, padding="max_length", truncation=True)
    else:
        # no need for padding
        tokenized_example = tokenizer(example)
    tokenized_example["labels"] = tokenized_example["input_ids"].copy()
    return tokenized_example


def generate_format(data_point, generate_user_prompt=False):
    """Function to generate the input format for training the model. If generate_user_prompt is set to True,
    it generates the input only."""
    if not generate_user_prompt:
        if data_point["input"]:
            full_prompt = data_point["instruction"] + "\n\n" + data_point["input"] + "\n\n" + data_point["output"] + "\n" + "<|endoftext|>"
        else:
            full_prompt = data_point["instruction"] + "\n\n" + data_point["output"] + "\n" + "<|endoftext|>"
    else:
        if data_point["input"]:
            full_prompt = data_point["instruction"] + "\n\n" + data_point["input"]
        else:
            full_prompt = data_point["instruction"]

    return full_prompt


def generate_and_tokenize_prompt(data_point):
    """Function to generate the specific training input format and tokenize it. If TRAIN_ON_INPUT set to False, the labels are masked
    to calculate the loss on the output only."""
    full_prompt = generate_format(data_point)
    tokenized_full_prompt = tokenize_function(full_prompt, single_texts=SINGLE_TEXTS)

    if not TRAIN_ON_INPUT:
        user_prompt = generate_format(data_point, generate_user_prompt=True)
        tokenized_user_prompt = tokenize_function(user_prompt, single_texts=SINGLE_TEXTS)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

    return tokenized_full_prompt


def single_texts(examples):
    """Function to create single data points and to pass them separately in training. Makes training slower since
    we pad them up to block size."""
    result = examples
    result["labels"] = examples["input_ids"].copy()
    return result


def group_texts(examples):
    "Function to concatenate all data points up to block size."
    # concatenate all texts
    concatenated_examples = {
        k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # split by chunks of block size
    result = {
        k: [t[i: i + BLOCK_SIZE]
            for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result


# data import and processing
base_path = DATA_PATH
# eval dataset only for callback
# raw_data = load_dataset("csv", data_files={"train": base_path + "train.csv", "validation": base_path + "test.csv", "eval": base_path + "eval.csv"})

raw_data = load_dataset("json", data_files=base_path + "large-data.json")

columns = raw_data["train"].column_names

raw_data = raw_data["train"].train_test_split(test_size=TEST_SIZE, shuffle=True, seed=42)

eval_data = list(map(lambda x: generate_format(x, generate_user_prompt=True), raw_data["test"].to_list()))

data = raw_data.map(generate_and_tokenize_prompt, remove_columns=columns)
if SINGLE_TEXTS:
    data = data.map(single_texts, batched=True)
else:
    data = data.map(group_texts, batched=True)


# training begins here
model.gradient_checkpointing_enable()
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    args=training_args,
    data_collator=transformers.default_data_collator,
    callbacks=[SampleCallback],
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.save_pretrained(OUTPUT_DIR)
