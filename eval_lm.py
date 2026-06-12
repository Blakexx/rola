"""Evaluate a RoLA checkpoint with lm-evaluation-harness (canonical zero-shot eval).
RoLA is AutoModel-registered (via `import rola_hf`), so lm_eval's HFLM loads it directly.

  python eval_lm.py --pretrained lm_runs/<run> --tasks lambada_openai,hellaswag,piqa,arc_easy
  # recall suite: --tasks swde,fda,squad_completion   (Based/Zoology info-extraction)
"""
import argparse, json
import rola_hf  # registers RoLA with HF AutoClasses so HFLM can load it
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True)
    p.add_argument("--tasks", default="lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande")
    # NOT "auto": on WSL2 the accelerate batch-finder probes a huge batch, OOMs at the
    # lm_head (vocab 50257), and the OOM surfaces as cudaErrorUnknown — a string the finder
    # doesn't recognize as OOM, so it dies instead of halving. Use an explicit batch.
    p.add_argument("--batch_size", default="8")
    # Match the training context (block_size=1024). The model's config max_position_embeddings
    # is 2048, but it trained at 1024; lm-eval would otherwise roll at 2048 and the
    # linear/recurrent head degrades out-of-distribution past its trained context.
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    bs = int(a.batch_size) if str(a.batch_size).isdigit() else a.batch_size
    lm = HFLM(pretrained=a.pretrained, backend="causal", batch_size=bs, max_length=a.max_length)
    res = simple_evaluate(model=lm, tasks=a.tasks.split(","), limit=a.limit)
    rows = res["results"]
    print("=== lm-eval results ===")
    for task, m in rows.items():
        print(f"  {task}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()
                                         if isinstance(v, (int, float))))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
