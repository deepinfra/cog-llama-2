# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from pydantic import BaseModel
import subprocess
import time
import requests
import sys

MODEL = "weights/llama-2-70b-chat.ggmlv3.q3_K_M.bin"

# for big (35B+ models)
EXTRA = "-gqa 8 -eps 1e-5 -ngl 1000".split(' ')
# for small (13B- models)
# EXTRA = "-ngl 1000".split(' ')

SELF_BUILD = True
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
        answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
         that your responses are socially unbiased and positive in nature.

         If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
                 correct. If you don't know the answer to a question, please don't share false information."""

class Response(BaseModel):
    generated_text: str
    generated_tokens: int

class Predictor(BasePredictor):

    def setup(self):
        if SELF_BUILD:
            # clean every time
            subprocess.run('rm -rf llama.cpp server', shell=True)
            subprocess.run('git clone https://github.com/ggerganov/llama.cpp', shell=True)
            subprocess.run('cd llama.cpp && LLAMA_CUBLAS=1 make -j && cp server ../', shell=True)
        
        self.port = 8080
        self.proc = subprocess.Popen(
                ["./server", "-m", MODEL, "--port", str(self.port)] + EXTRA)
        self.base_url = f"http://127.0.0.1:{self.port}"

        while True:
            time.sleep(1)
            try:
                res = requests.get(self.base_url)
                if res.status_code == 200:
                    break
            except:
                pass
            print('waiting for server')

    def predict(
        self,
        # system: str = Input(description="system instructions for model", default=DEFAULT_SYSTEM_PROMPT),
        prompt: str = Input(description="model prompt"),
        max_length: int = Input(default=100, description="maximum number of tokens to generate, -1 for unlimited"),
        top_k: int = Input(default=40, description="sample from top k options for each token, 0 to disable", ge=0),
        top_p: float = Input(default=0.95, description="sample from top options totalling at least p (0-1), 1.0 to disable", le=1.0, ge=0.0),
        temperature: float = Input(default=0.8, description="how much to prioritize better matching tokens, 1.0 is neutral, > 1.0 choose worse-prob tokens"),
        seed: int = Input(default=-1, description="seed for randomness, -1 for random"),
    ) -> Response:
        """Run a single prediction on the model"""
        req = {
            'prompt': prompt,
            'n_predict': max_length,
            'top_k': top_k,
            'top_p': top_p,
            'seed': seed,
            'temp': temperature,
        }

        res = requests.post(
            f"{self.base_url}/completion",
            json=req,
        )
        assert res.status_code == 200
        # print(f"RES {res.json()}")
        obj = res.json()
        return Response(
            generated_text=f'{prompt} {obj["content"]}',
            generated_tokens=obj['timings']['predicted_n'],
        )

    def conv_to_prompt(self, conv: list[str]) -> list[str|int]:
        """The output is compatible with the new server API that accepts a list
        of strings/tokens. This is necessary to replicate the exact
        conversation token sequence used during llama 2 training"""
        # conv = conv.split(',')
        assert len(conv) % 2 == 1, "expected odd number of conversation items"
        prompt: list[str|int] = []
        for i, (q, a) in enumerate(zip(conv[::2], conv[1::2])):
            if i == 0:
                q = f"<<SYS>>\n{system}\n<</SYS>>\n\n{q}"

            prompt.append(1) # bos
            prompt.append(f"[INST] {q} [/INST] {a} ")
            prompt.append(2) # eos

        prompt.append(1)
        prompt.append(f"[INST] {conv[-1]} [/INST]")

        # drop first bos
        assert prompt.pop(0) == 1
        return prompt
