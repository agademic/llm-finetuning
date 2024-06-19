from fastapi import FastAPI
from classes import GogohiGPTRequest, GogohiGPTResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftConfig, PeftModel

app = FastAPI()

base_model_id = "model/"  # GPT-Neo-125M in this example

if torch.cuda.is_available():
    device = torch.device('cuda')

    print(f"There are {torch.cuda.device_count()} GPUs available.")
    print(f"GPU {torch.cuda.get_device_name(0)} will be used.")
else:
    print("No GPU available, using CPU instead.")
    device = torch.device('cpu')

# code to load the model and merge with adapter if we trained one
# @app.on_event("startup")
# async def load_model():
#     global model, tokenizer
#     config = PeftConfig.from_pretrained(peft_model_id)
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_id, return_dict=True, load_in_8bit=True, device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(base_model_id)

#     # Load the Lora model
#     model = PeftModel.from_pretrained(model, peft_model_id, device_map={'': 0})


@app.on_event("startup")
async def load_model():
    print("loading model...")
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)


@app.get("/hello")
async def hello():
    return {"message": "Hello There!"}


@app.get("/")
async def root():
    return {"message": "Application running!"}


@app.post("/generate")
async def gpt(request: GogohiGPTRequest):
    global model, tokenizer
    prompt = request.prompt
    temperature = request.temperature
    top_k = request.top_k
    top_p = request.top_p
    max_new_tokens = request.max_new_tokens
    do_sample = request.do_sample

    batch = tokenizer(prompt, return_tensors="pt").to(device)
    print(request)
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample)
    # generate(request, model, tokenizer, device)

    response = GogohiGPTResponse(text=tokenizer.decode(output_tokens[0]))
    return response
