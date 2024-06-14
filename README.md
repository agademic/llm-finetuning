# llm-finetuning
This repository contains examples on how to fine-tune (more specific: instruction-tune) different generative large language models.

## Supervised Fine-Tuning
In this repo, we are going to instruction-tune a large language model via Supervised Fine-Tuning (SFT). With this method, we pass thousands of examples to the model on how we want it to behave on certain instructions.

There are multiple techniques to fine-tune a model. The most straightforward method is to further train the model and all of its weighths. This is computationally very expensive as each layer needs to be loaded into memory to further train the whole model.

A more feasible method is to inject adapters (small feed-forward networks) into each transformer block of the LLM and train only these adapters. This lowers the memory footprint, as we do not need to backpropagate trough all weights of the model but only of a few new weights.

### Low-Rank Adaptation
LoRA (Low-Rank Adaptation) of large language models takes the idea of injecting adapter layers into the model a step further. Instead of injecting small feed-forward networks, we add a decomposed weigth matrix to each forward pass within the model (which are two low-rank matrices). These matrices are learned during training.

<p align="center">
  <img src="img/image.png" alt=""/>
  <figcaption>Fig.1 - LoRA: Low-Rank Adaptation of Large Language Models (2021, Edward J. Hu et al.)</figcaption>
</p>

The parameters, we can choose for LoRA are its alpha value and its rank.
- alpha: this is the scaling factor of LoRA. With this value we can control the amount of change we want to add to the original weights. This is a trade-off between the previous general knowledge learned by the model and the fine-tuned new task.
- rank: value to which extent we want to decompose the matrices. Typically a value between 1 and 64. We choose a higher value if our new task is substantially different.

### Datasets
To fine-tune our (very) general model, we need to choose an (or multiple) appropriate dataset(s). Here, we are going to use the OG datasets `alpaca-self-instruct`, `natural-instruction` and `unnatural-instructions`. With these so-called instruction datasets, we are going to "show" the model how to behave when we prompt it to perform a task.

In the folder `data_scripts` you can find the processing scripts for each of the mentioned datasets. In order to create a large dataset from them, we need to standardize the datasets first. The individual scripts for each dataset aim to bring each dataset to the following format:

```json
{
    "instruction": "...",
    "input": "...",
    "output": "..."
}
```

After standardizing the individual datasets, we concatenate all datasets into a single file and then sample train, test and dev splits.

Finally, we will bring all data into a format for fine-tuning. In the final format, we will join all fields from above into single strings containing instructions, input (where applicable) and output.