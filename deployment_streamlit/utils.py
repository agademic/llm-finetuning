import requests
import torch
from classes import GogohiGPTRequest


def send_request(input_data: GogohiGPTRequest):
    url = "http://127.0.0.1:8000/generate"
    response = requests.post(url, json=input_data.dict())

    return response


def generate(input_data: GogohiGPTRequest, model, tokenizer, device):
    # Set the maximum length of the generated text
    prompt = input_data.prompt
    max_new_tokens = input_data.max_new_tokens

    # Tokenize the prompt and convert it to a tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

    # Generate text word by word
    for i in range(max_new_tokens):
        # Generate the next token from the model
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=i + 1
        )
        # Get the last generated token
        token = output[:, -1]
        # Convert the token ID to its corresponding token string
        word = tokenizer.decode(token.item())
        # Print the generated word
        print(word, end=' ')
        # Add the generated token to the input prompt for the next iteration
        input_ids = torch.cat([input_ids, token.unsqueeze(0)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
