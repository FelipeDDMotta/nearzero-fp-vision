"""Generate distill/finetune_qlora.ipynb (kept as code so it diffs cleanly)."""
import json
from pathlib import Path

REPO = "https://github.com/FelipeDDMotta/nearzero-fp-vision.git"
cells = []


def md(s: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# NearZeroFP Vision: distill the Claude verifier into Qwen2.5-VL-3B (QLoRA) and benchmark it with vLLM

**Kaggle settings:** Accelerator = *GPU T4 x2*, Internet = *on*. Add a Kaggle Secret named `HF_TOKEN` (a Hugging Face write token).

What this does:
1. Clones the repo and exports the agent's decisions as a chat dataset. One camera is held out.
2. QLoRA fine-tune (4-bit, r=16, language layers only) with Unsloth on GPU 0.
3. Merges the adapter and pushes fp16 weights to the Hugging Face Hub (used by the Modal vLLM endpoint).
4. Serves the base and fine-tuned models with **vLLM** on GPU 1 and scores both on the held-out camera.
5. Uploads the predictions next to the model so `distill/bench_report.py` can build the site's table.
""")

code(f"""
!pip install -q "unsloth==2026.11.7" anthropic httpx
!pip uninstall -y -q xformers   # T4 has no FlashAttention; xformers needs seq len % 4 == 0. Fall back to SDPA.
!git clone -q {REPO} && echo cloned
%cd nearzero-fp-vision
HOLDOUT = "street-night-rain,walkway-dusk"   # cameras/conditions never seen in training
HF_REPO = "FelipeDdM/nearzero-fp-verifier-qwen2.5-vl-3b"
BASE = "unsloth/Qwen2.5-VL-3B-Instruct"
!python distill/export_vlm_dataset.py --holdout {{HOLDOUT}}
""")

code("""
import json, random
from collections import Counter
from pathlib import Path
from PIL import Image

D = Path("distill/data")
rows = lambda split: [json.loads(l) for l in (D / f"{split}.jsonl").read_text().splitlines() if l.strip()]
train_rows, test_rows = rows("train"), rows("test")

# Light rebalancing: oversample minority verdicts up to half of the majority class.
by = {}
for r in train_rows:
    by.setdefault(json.loads(r["target"])["verdict"], []).append(r)
major = max(len(v) for v in by.values())
random.seed(0)
for k, v in by.items():
    while len(v) < major // 2:
        v.append(random.choice(v[: len(v)]))
train_rows = [r for v in by.values() for r in v]
random.shuffle(train_rows)

def to_conv(r):
    imgs = [Image.open(D / p).convert("RGB") for p in r["images"]]
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": imgs[0]}, {"type": "image", "image": imgs[1]},
                                      {"type": "text", "text": r["prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": r["target"]}]},
    ]}

train = [to_conv(r) for r in train_rows]
print("train", len(train), Counter(json.loads(r["target"])["verdict"] for r in train_rows))
print("test ", len(test_rows), Counter(json.loads(r["target"])["verdict"] for r in test_rows))
""")

code("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # train on GPU 0, keep GPU 1 free for vLLM
import torch
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(BASE, load_in_4bit=True, use_gradient_checkpointing="unsloth")
model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
)
""")

code("""
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train,
    args=SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=8,  # bs=1: no padding warmup_steps=5,
        num_train_epochs=2, learning_rate=2e-4, fp16=True, bf16=False,
        logging_steps=5, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        seed=3407, output_dir="outputs", report_to="none",
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
    ),
)
stats = trainer.train()
Path("distill/out").mkdir(parents=True, exist_ok=True)
Path("distill/out/train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
print(stats)
""")

code("""
# Sanity check on a few held-out samples with the in-memory model.
FastVisionModel.for_inference(model)
for r in test_rows[:3]:
    conv = to_conv(r)["messages"][:1]
    inputs = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_dict=True,
                                           tokenize=True, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=120, temperature=0.0, do_sample=False)
    print(r["id"], "target:", r["target"][:80])
    print("   pred:", tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:200])
""")

code("""
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(HF_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
print("pushed", HF_REPO)
""")

code("""
# vLLM in a fresh process on GPU 1 (the training process still holds GPU 0).
del trainer; import gc; gc.collect(); torch.cuda.empty_cache()
!pip install -q vllm==0.33.0
!CUDA_VISIBLE_DEVICES=1 python distill/bench_vllm.py --model {BASE} --tag base
!CUDA_VISIBLE_DEVICES=1 python distill/bench_vllm.py --model ./merged --tag lora
""")

code("""
from huggingface_hub import HfApi
HfApi(token=HF_TOKEN).upload_folder(folder_path="distill/out", path_in_repo="bench", repo_id=HF_REPO)
for f in sorted(Path("distill/out").glob("pred_*.json")):
    d = json.loads(f.read_text())
    acc = sum(p["pred"] == p["target"] for p in d["preds"]) / max(1, d["n"])
    print(f.name, "acc", round(acc, 3), "ms/sample", d["per_sample_ms"], "tok/s", d["output_tokens_per_s"])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("finetune_qlora.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
