"""Serve the distilled verifier (Qwen-VL + merged LoRA) with vLLM on Modal.

Scale-to-zero GPU endpoint with an OpenAI-compatible API, protected by a bearer token.

Setup (once):
  pip install modal && modal setup
  modal secret create nearzero-vlm VLLM_API_KEY=<random-string>
  STUDENT_MODEL=FelipeDdM/nearzero-fp-verifier-qwen2.5-vl-3b modal deploy serving/modal_app.py
Then set on Render: OPEN_VLM_URL=<printed url>, OPEN_VLM_TOKEN=<same random string>.
"""
import os
import subprocess

import modal

MODEL = os.environ.get("STUDENT_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
SERVED_NAME = "nearzero-verifier"
GPU = os.environ.get("MODAL_GPU", "L4")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.33.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "STUDENT_MODEL": MODEL})
)
hf_cache = modal.Volume.from_name("nearzero-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("nearzero-vllm-cache", create_if_missing=True)

app = modal.App("nearzero-vlm")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=300,  # stay warm 5 min after the last request, then scale to zero
    timeout=900,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("nearzero-vlm")],
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    cmd = [
        "vllm", "serve", os.environ["STUDENT_MODEL"],
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", "4096",
        "--limit-mm-per-prompt", '{"image": 2}',
        "--gpu-memory-utilization", "0.90",
        "--dtype", "half",
    ]
    subprocess.Popen(cmd)
