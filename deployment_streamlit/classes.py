from pydantic import BaseModel


class GogohiGPTRequest(BaseModel):
    prompt: str
    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 2
    do_sample: bool = False
    max_new_tokens: int = 50


class GogohiGPTResponse(BaseModel):
    text: str
