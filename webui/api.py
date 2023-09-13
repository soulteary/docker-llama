from typing import Optional, Union, List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

app_env = {
    "app": app,
    "generator": None # LLaMA
}

"""
body: {
    "prompts": ["text1", "text2],
    "max_gen_len": 1024, # default
    "temperature": 0.8, # default
    "top_p": 0.95, # default
    "repetition_penalty": { # default
        "range": 1024,
        "slope": 0,
        "value": 1.15
    }
}
"""

class RepetitionPenalty(BaseModel):
    range: Optional[int] = 1024
    slope: Optional[float] = 0
    value: Optional[float] = 1.15

class GenerateParam(BaseModel):
    prompts: Union[List[str], str]
    max_gen_len: Optional[int] = 1024
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[RepetitionPenalty] = RepetitionPenalty()

@app.post('/generate')
async def generate(params: GenerateParam):
    if len(params.prompts) == 0:
        return {
            "error": -1,
            "msg": "There are prompts should be provided."
        }

    if type(params.prompts) is not list:
        params.prompts = [params.prompts]

    results = app_env["generator"].generate(
        prompts = params.prompts,
        max_gen_len = params.max_gen_len,
        temperature = params.temperature,
        top_p = params.top_p,
        repetition_penalty_range = params.repetition_penalty.range,
        repetition_penalty_slope = params.repetition_penalty.slope,
        repetition_penalty = params.repetition_penalty.value
    )

    return {
        "results": results
    }


def get_args():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt_dir", type=str, default="./models/7B")
    # parser.add_argument("--tokenizer_path", type=str, default="./models/tokenizer.model")
    # return parser.parse_args()

    return {
        "ckpt_dir": "./models/7B",
        "tokenizer_path": "./models/tokenizer.model"
    }