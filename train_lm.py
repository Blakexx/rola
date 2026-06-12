"""Single-GPU RoLA LM training via the canonical HF Trainer + datasets (run_clm recipe):
load corpus -> GPT-2 BPE tokenize -> group into block_size -> Trainer (loop/optim/sched/
ckpt/eval are HF's, not hand-rolled). RoLA is an AutoModel-registered HF model (rola_hf),
so it plugs straight in; the saved checkpoint runs in lm-evaluation-harness as-is.

  python train_lm.py --rola_instance rola-rla-asym --hidden_size 512 --num_hidden_layers 12 \
     --num_states 16 --d_qk 16 --d_v 16 --num_heads 8 --max_steps 20000

For SCALED multi-GPU FineWeb pretraining use flame (torchtitan) instead; this is the cheap
single-GPU path (WikiText-103 first).
"""
import argparse, math
from itertools import chain
import rola_hf  # registers RoLA with HF AutoClasses
from rola_hf import RoLAConfig
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling)


FINEWEB = "HuggingFaceFW/fineweb-edu"  # natural text -> matches lm-eval's detokenized format


def build_dataset(name, config, block_size, tokenizer, num_proc=8, max_train_samples=None):
    # Friendly alias: --dataset fineweb-edu -> the canonical FineWeb-Edu sample.
    if name in ("fineweb", "fineweb-edu"):
        name = FINEWEB
        config = config if (config or "").startswith("sample") else "sample-10BT"
    # Slice the train split for calibration / token-budgeted runs (avoids the full
    # 27GB download: HF fetches only the shards the slice touches).
    train_split = f"train[:{max_train_samples}]" if max_train_samples else "train"
    from datasets import DatasetDict
    train_raw = load_dataset(name, config, split=train_split)
    try:                                   # WikiText etc. ship a validation split
        val_raw = load_dataset(name, config, split="validation")
    except Exception:                      # FineWeb-Edu is train-only -> carve a tail
        sp = train_raw.train_test_split(test_size=min(2000, max(2, len(train_raw) // 50)), seed=0)
        train_raw, val_raw = sp["train"], sp["test"]
    raw = DatasetDict(train=train_raw, validation=val_raw)
    def tok(ex): return tokenizer(ex["text"])
    cols = raw["train"].column_names
    t = raw.map(tok, batched=True, remove_columns=cols, num_proc=num_proc, desc="tokenize")
    def group(ex):
        # Canonical run_clm group_texts: concatenate + chunk EVERY key consistently.
        cat = {k: list(chain(*ex[k])) for k in ex.keys()}
        n = (len(cat["input_ids"]) // block_size) * block_size
        res = {k: [t[i:i + block_size] for i in range(0, n, block_size)] for k, t in cat.items()}
        res["labels"] = [x[:] for x in res["input_ids"]]
        return res
    return t.map(group, batched=True, num_proc=num_proc, desc=f"group {block_size}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rola_instance", default="rola-rla-asym")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_states", type=int, default=16)
    p.add_argument("--d_qk", type=int, default=16)
    p.add_argument("--d_v", type=int, default=16)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="slice train to N docs (calibration / token-budgeted runs)")
    a = p.parse_args()
    out = a.out_dir or f"lm_runs/{a.rola_instance}-d{a.hidden_size}L{a.num_hidden_layers}"

    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    ds = build_dataset(a.dataset, a.dataset_config, a.block_size, tok, a.num_proc, a.max_train_samples)
    cfg = RoLAConfig(rola_instance=a.rola_instance, num_states=a.num_states, d_qk=a.d_qk,
                     d_v=a.d_v, num_heads=a.num_heads, hidden_size=a.hidden_size,
                     num_hidden_layers=a.num_hidden_layers, vocab_size=len(tok),
                     bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    model = AutoModelForCausalLM.from_config(cfg)
    print(f"[model] {a.rola_instance} {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    args = TrainingArguments(
        output_dir=out, max_steps=a.max_steps, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=a.warmup,
        weight_decay=a.weight_decay, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50, eval_strategy="steps", eval_steps=a.eval_steps,
        # No mid-training checkpoints: the Trainer's default safetensors save crashes on sym's
        # tied routers (read_router==write_router is a shared tensor safetensors refuses). We
        # save once at the end as .bin (safe_serialization=False) which allows shared tensors.
        save_strategy="no",
        report_to="none", dataloader_num_workers=4)
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["validation"],
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    metrics = trainer.evaluate()
    print(f"DONE eval_loss={metrics['eval_loss']:.4f} ppl={math.exp(min(metrics['eval_loss'],20)):.2f}", flush=True)
    model.save_pretrained(out, safe_serialization=False); tok.save_pretrained(out)  # .bin: allows sym's shared routers


if __name__ == "__main__":
    main()
