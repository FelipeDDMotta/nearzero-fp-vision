"""Generate distill/finetune_v2.ipynb: leak-free split, free-label data, vision layers unfrozen."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# NearZeroFP Vision: distillation v2

**Kaggle:** Settings → Accelerator *GPU T4 x2*, Internet *on*; Add-ons → Secrets: `HF_TOKEN` (write).

Changes from v1 (which tied zero-shot at 70% on n=23, with a leaky split):
- **Split by source video:** the test videos (street + walkway, every synthetic variant) never appear in training.
- **More data at zero labelling cost:** teacher + human labels, plus free labels (detector-confident tracks,
  no-target-scene detections, synthetic background negatives).
- **No numeric shortcuts in the prompt:** no detector score or track length.
- **Vision layers adapted too**, 3 epochs, lr 1e-4.
- Headline metric: accuracy on the **hard** items of the held-out videos.
""")

code("""
!pip install -q "unsloth==2026.11.7" httpx
!pip uninstall -y -q xformers   # T4: avoid the xformers attn_bias alignment error (see AGENT_LOG #3)
!rm -rf nearzero-fp-vision && git clone -q https://github.com/FelipeDDMotta/nearzero-fp-vision.git && echo cloned
%cd nearzero-fp-vision
HF_REPO = "FelipeDdM/nearzero-fp-verifier-qwen2.5-vl-3b-v2"
BASE = "unsloth/Qwen2.5-VL-3B-Instruct"
""")

code("""
import json, random
from collections import Counter
from pathlib import Path
from PIL import Image

D = Path("distill/data_v2")
rows = lambda s: [json.loads(l) for l in (D / f"{s}.jsonl").read_text().splitlines() if l.strip()]
train_rows, test_rows = rows("train"), rows("test")
random.seed(0)
# Keep the free 'easy' bulk from drowning the hard examples: cap each easy bucket.
hard = [r for r in train_rows if r["difficulty"] == "hard"]
easy_pos = [r for r in train_rows if r["difficulty"] == "easy" and r["label"] == "true_positive"]
easy_neg = [r for r in train_rows if r["difficulty"] == "easy" and r["label"] == "false_positive"]
random.shuffle(easy_pos); random.shuffle(easy_neg)
train_rows = hard + easy_pos[:250] + easy_neg[:200]
random.shuffle(train_rows)

def to_conv(r):
    imgs = [Image.open(D / p).convert("RGB") for p in r["images"]]
    return {"messages": [
        {"role": "user", "content": [{"type": "image", "image": imgs[0]}, {"type": "image", "image": imgs[1]},
                                      {"type": "text", "text": r["prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": r["target"]}]}]}

train = [to_conv(r) for r in train_rows]
print("train", len(train), Counter((r["difficulty"], r["label"]) for r in train_rows))
print("test ", len(test_rows), Counter((r["difficulty"], r["label"]) for r in test_rows))
""")

code("""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(BASE, load_in_4bit=True, use_gradient_checkpointing="unsloth")
model = FastVisionModel.get_peft_model(
    model, finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", random_state=3407,
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
        per_device_train_batch_size=1, gradient_accumulation_steps=8, warmup_steps=10,
        num_train_epochs=3, learning_rate=1e-4, fp16=True, bf16=False,
        logging_steps=10, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
        seed=3407, output_dir="outputs_v2", report_to="none",
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
    ),
)
stats = trainer.train()
Path("distill/out").mkdir(parents=True, exist_ok=True)
Path("distill/out/train_log_v2.json").write_text(json.dumps(trainer.state.log_history, indent=1))
print(stats)
""")

code("""
from kaggle_secrets import UserSecretsClient
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
model.save_pretrained_merged("merged_v2", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged(HF_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
print("pushed", HF_REPO)
""")

code("""
del trainer; import gc; gc.collect(); torch.cuda.empty_cache()
!pip install -q vllm==0.33.0
!CUDA_VISIBLE_DEVICES=1 python distill/bench_v2.py --model {BASE} --tag v2_base
!CUDA_VISIBLE_DEVICES=1 python distill/bench_v2.py --model ./merged_v2 --tag v2_lora
""")

code("""
from huggingface_hub import HfApi
HfApi(token=HF_TOKEN).upload_folder(folder_path="distill/out", path_in_repo="bench", repo_id=HF_REPO)
print("uploaded results to", HF_REPO)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path(__file__).with_name("finetune_v2.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote finetune_v2.ipynb")
