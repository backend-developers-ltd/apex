import asyncio
import os
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from schema import ChatRequest, LogitsRequest
from vllm_llm import ReproducibleVLLM

MODEL_PATH = os.getenv("MODEL_PATH")


@dataclass
class Tokenizer:
    bos_token: str
    eos_token: str


class ReproducibleVllmApp:
    def __init__(self):
        self.llm = ReproducibleVLLM(model_id=MODEL_PATH)

        self.app = FastAPI()

        self.app.get("/health")(self.health)
        self.app.get("/tokenizer")(self.tokenizer)
        self.app.post("/terminate")(self.terminate)
        self.app.post("/generate")(self.generate)
        self.app.post("/generate_logits")(self.generate_logits)

    async def health(self):
        return JSONResponse(status_code=200, content={"status": "OK"})

    async def tokenizer(self):
        try:
            bos_token = self.llm.tokenizer.bos_token
            eos_token = self.llm.tokenizer.eos_token
            return JSONResponse(status_code=200, content={"bos_token": bos_token, "eos_token": eos_token})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def generate(self, request: ChatRequest):
        try:
            result = await self.llm.generate(
                messages=[m.model_dump() for m in request.messages],
                sampling_params=request.sampling_parameters.model_dump(),
                seed=request.seed,
                continue_last_message=request.continue_last_message,
            )
            return {"result": result}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def generate_logits(self, request: LogitsRequest):
        try:
            logits, prompt = await self.llm.generate_logits(
                messages=[m.model_dump() for m in request.messages],
                top_logprobs=request.top_logprobs,
                sampling_params=request.sampling_parameters.model_dump(),
                seed=request.seed,
                continue_last_message=request.continue_last_message,
            )
            return {"logits": logits, "prompt": prompt}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def terminate(self, request: Request):
        response = JSONResponse(status_code=200, content={"status": "OK"})
        asyncio.create_task(self.shutdown_after_delay(1))
        return response

    async def shutdown_after_delay(self, delay_seconds: int = 1):
        await asyncio.sleep(delay_seconds)
        print("Terminating server on request.")
        os._exit(0)

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    server = ReproducibleVllmApp()
    server.run()
