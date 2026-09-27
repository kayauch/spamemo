from __future__ import annotations

import os
import time
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3-4B")
MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "384"))
TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))

app = FastAPI(title="Local Spatial LLM", version="0.1.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None


class LocalQwen:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_seconds: float | None = None
        self.load_peak_gib: float | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.load_peak_gib = torch.cuda.max_memory_allocated() / 2**30

    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int,
        temperature: float,
        json_only: bool,
    ) -> tuple[str, dict[str, float]]:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        rendered = [{"role": m.role, "content": m.content} for m in messages]
        if json_only:
            rendered.insert(
                0,
                {
                    "role": "system",
                    "content": "Return only one valid JSON object. Do not use markdown fences or commentary.",
                },
            )

        template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            inputs = self.tokenizer.apply_chat_template(
                rendered,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                rendered,
                **template_kwargs,
            )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        do_sample = temperature > 0.0
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generated = output[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        metrics = {
            "generation_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "input_tokens": float(inputs["input_ids"].shape[-1]),
            "output_tokens": float(generated.shape[-1]),
        }
        return text, metrics


engine = LocalQwen(MODEL_ID)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": engine.model is not None,
        "load_seconds": engine.load_seconds,
        "load_peak_gib": engine.load_peak_gib,
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> dict:
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages required")

    max_new_tokens = min(
        request.max_tokens or MAX_NEW_TOKENS,
        1024,
    )
    temperature = (
        TEMPERATURE
        if request.temperature is None
        else max(0.0, min(request.temperature, 1.5))
    )
    json_only = bool(
        request.response_format
        and request.response_format.get("type") == "json_object"
    )

    text, metrics = engine.generate(
        request.messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        json_only=json_only,
    )

    return {
        "id": f"local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(metrics["input_tokens"]),
            "completion_tokens": int(metrics["output_tokens"]),
            "total_tokens": int(
                metrics["input_tokens"] + metrics["output_tokens"]
            ),
        },
        "metrics": metrics,
    }
