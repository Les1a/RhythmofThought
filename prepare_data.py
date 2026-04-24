#!/usr/bin/env python
"""
prepare_data.py — Unified train/eval data preparation for GRPO/TGRPO/HRPO/THRPO.

Single entry point that replaces the four legacy scripts
(prepare_eval_data.py, prepare_math_data.py, prepare_mmlu_train_data.py,
prepare_rag_train_data.py) with a task/stage-driven CLI matching the style
of the run_*_all.sh launcher --tasks flag.

Usage:
  # Default: prepare train + eval for all tasks
  python prepare_data.py

  # Only MATH (train and test splits)
  python prepare_data.py --tasks math

  # Only RAG eval with BM25 retrieval for NQ/TQ/Bamboogle
  python prepare_data.py --tasks rag --stage eval --with-retrieval

  # Force re-prep of MMLU train
  python prepare_data.py --tasks mmlu --stage train --force

Per-task outputs
----------------
  gsm8k : no-op (openai/gsm8k auto-downloads at runtime)
  math  : ../MATH/{train,test}/<subject>/<idx>.json
  mmlu  : train → ../MMLU_Train_Merged/ (HF save_to_disk)
          eval  → no-op (TIGER-Lab/MMLU-STEM + allenai/ai2_arc auto-download)
  rag   : train → ../RAG_Train_Merged/ (HF save_to_disk)
          eval  → ../RAG_Eval/{HotpotQA,2Wiki,NQ,TQ,Bamboogle}_Eval
                  (NQ/TQ/Bamboogle get empty contexts unless --with-retrieval)

Retrieval note
--------------
  The HRPO paper uses E5 embeddings + Wikipedia 2020 ANN search for NQ/TQ/Bamboogle
  contexts. --with-retrieval here is a lightweight BM25 approximation over
  HotpotQA+2Wiki paragraphs (max 50k) for convenience. If rank_bm25 is missing or
  retrieval fails mid-run, we fall back to closed-book (empty contexts) with a
  warning and continue — no shell-side retry needed.
"""
import argparse
import json
import os
import shutil
import sys
import traceback

from datasets import Dataset, load_dataset


# ============================================================================
# Paths
# ============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MATH_DIR = os.path.join(_SCRIPT_DIR, "..", "MATH")
MMLU_TRAIN_DIR = os.path.join(_SCRIPT_DIR, "..", "MMLU_Train_Merged")
RAG_TRAIN_DIR = os.path.join(_SCRIPT_DIR, "..", "RAG_Train_Merged")
RAG_EVAL_DIR = os.path.join(_SCRIPT_DIR, "..", "RAG_Eval")

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

RETRIEVAL_DATASETS = {
    "nq": ("nq", "NQ_Eval"),
    "triviaqa": ("triviaqa", "TQ_Eval"),
    "bamboogle": ("bamboogle", "Bamboogle_Eval"),
}


# ============================================================================
# Logging helpers
# ============================================================================

def log(task, msg):
    """Emit a one-line task-scoped log message."""
    print(f"[{task}] {msg}", flush=True)


def log_section(title):
    """Print a section divider for long-running preparation steps."""
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


# ============================================================================
# BM25 retriever — graceful fallback if rank_bm25 is missing
# ============================================================================

class BM25Retriever:
    """Lightweight BM25 retriever over HotpotQA+2Wiki paragraphs.

    Raises ImportError on construction if rank_bm25 is not installed, so
    callers can catch and fall back to closed-book mode.
    """

    def __init__(self):
        from rank_bm25 import BM25Okapi  # may raise ImportError
        self._BM25Okapi = BM25Okapi
        self.corpus = []
        self.index = None

    def build_index(self, max_paragraphs=50000):
        log("rag", f"Building BM25 index (max {max_paragraphs} paragraphs)...")
        seen_titles = set()
        corpus_tokenized = []

        def _ingest(title, parts):
            if title in seen_titles:
                return False
            seen_titles.add(title)
            passage = f'"{title}"\n' + " ".join(p.strip() for p in parts)
            self.corpus.append(passage)
            tokenized_text = (
                title.lower() + " " + " ".join(p.strip().lower() for p in parts)
            ).split()
            corpus_tokenized.append(tokenized_text)
            return True

        # HotpotQA (sentences)
        hotpot = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="dev")
        for sample in hotpot:
            if len(self.corpus) >= max_paragraphs:
                break
            ctx = (sample.get("metadata") or {}).get("context") or {}
            for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
                if len(self.corpus) >= max_paragraphs:
                    break
                _ingest(title, sents)

        # 2Wiki (content)
        if len(self.corpus) < max_paragraphs:
            wiki2 = load_dataset(
                "RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split="dev"
            )
            for sample in wiki2:
                if len(self.corpus) >= max_paragraphs:
                    break
                ctx = (sample.get("metadata") or {}).get("context") or {}
                for title, content in zip(ctx.get("title", []), ctx.get("content", [])):
                    if len(self.corpus) >= max_paragraphs:
                        break
                    _ingest(title, content)

        log("rag", f"  Corpus size: {len(self.corpus)} unique paragraphs")
        self.index = self._BM25Okapi(corpus_tokenized)
        log("rag", "  BM25 index built.")

    def retrieve(self, query, topk=10):
        if self.index is None:
            self.build_index()
        scores = self.index.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [self.corpus[i] for i in top_indices]


# ============================================================================
# GSM8K
# ============================================================================

def prepare_gsm8k(stage, force=False):
    log("gsm8k", f"stage={stage}: no prep required — openai/gsm8k auto-downloads at runtime")


# ============================================================================
# MATH
# ============================================================================

def _download_math_split(split, force=False):
    """Download one split of EleutherAI/hendrycks_math into ../MATH/<split>/<subject>/<idx>.json."""
    split_root = os.path.join(MATH_DIR, split)
    # Treat as done if every subject dir already has contents
    if not force and os.path.isdir(split_root):
        done = all(
            os.path.isdir(os.path.join(split_root, s)) and os.listdir(os.path.join(split_root, s))
            for s in MATH_SUBJECTS
        )
        if done:
            log("math", f"  [skip] ../MATH/{split}/ already populated for all subjects")
            return

    os.makedirs(split_root, exist_ok=True)
    total = 0
    for subject in MATH_SUBJECTS:
        out_dir = os.path.join(split_root, subject)
        os.makedirs(out_dir, exist_ok=True)
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        except Exception as e:
            log("math", f"  [WARN] Failed to load {subject}/{split}: {e}")
            continue
        for idx, sample in enumerate(ds):
            with open(os.path.join(out_dir, f"{idx}.json"), "w") as f:
                json.dump({"problem": sample["problem"], "solution": sample["solution"]}, f)
        total += len(ds)
    log("math", f"  Downloaded MATH {split}: {total} problems")


def prepare_math(stage, force=False):
    if stage in ("train", "all"):
        log("math", "stage=train: downloading EleutherAI/hendrycks_math train splits")
        _download_math_split("train", force=force)
    if stage in ("eval", "all"):
        log("math", "stage=eval: downloading EleutherAI/hendrycks_math test splits")
        _download_math_split("test", force=force)
        log("math", "  note: HuggingFaceH4/MATH-500 used by eval_math.py auto-downloads at eval time")


# ============================================================================
# MMLU
# ============================================================================

def prepare_mmlu(stage, force=False):
    if stage in ("train", "all"):
        log("mmlu", "stage=train: preparing ../MMLU_Train_Merged from cais/mmlu:auxiliary_train")
        if os.path.isdir(MMLU_TRAIN_DIR) and not force:
            log("mmlu", f"  [skip] {MMLU_TRAIN_DIR} already exists")
        else:
            if force and os.path.isdir(MMLU_TRAIN_DIR):
                log("mmlu", f"  [force] removing {MMLU_TRAIN_DIR}")
                shutil.rmtree(MMLU_TRAIN_DIR)
            ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
            log("mmlu", f"  Loaded {len(ds)} raw examples; unwrapping nested 'train' column")
            questions, choices, answers = [], [], []
            for row in ds:
                inner = row["train"]
                questions.append(inner["question"])
                choices.append(inner["choices"])
                answers.append(inner["answer"])
            new_ds = Dataset.from_dict({
                "question": questions,
                "choices": choices,
                "answer": answers,
            })
            new_ds.save_to_disk(MMLU_TRAIN_DIR)
            log("mmlu", f"  Saved {len(new_ds)} examples → {MMLU_TRAIN_DIR}")

    if stage in ("eval", "all"):
        log("mmlu", "stage=eval: no prep — TIGER-Lab/MMLU-STEM and allenai/ai2_arc auto-download")


# ============================================================================
# RAG — training (SQuAD reformatted)
# ============================================================================

def prepare_rag_train(force=False):
    log("rag", "stage=train: preparing ../RAG_Train_Merged from rajpurkar/squad")
    if os.path.isdir(RAG_TRAIN_DIR) and not force:
        log("rag", f"  [skip] {RAG_TRAIN_DIR} already exists")
        return
    if force and os.path.isdir(RAG_TRAIN_DIR):
        log("rag", f"  [force] removing {RAG_TRAIN_DIR}")
        shutil.rmtree(RAG_TRAIN_DIR)

    ds = load_dataset("rajpurkar/squad", split="train")
    log("rag", f"  Loaded {len(ds)} raw examples")

    questions = list(ds["question"])
    contexts = [[c] for c in ds["context"]]
    golden_answers = [a["text"] for a in ds["answers"]]

    new_ds = Dataset.from_dict({
        "question": questions,
        "contexts": contexts,
        "golden_answers": golden_answers,
    })
    new_ds.save_to_disk(RAG_TRAIN_DIR)
    log("rag", f"  Saved {len(new_ds)} examples → {RAG_TRAIN_DIR}")


# ============================================================================
# RAG — eval datasets
# ============================================================================

def _convert_metadata_context(metadata_context, content_key):
    titles = metadata_context["title"]
    parts_lists = metadata_context[content_key]
    return [
        f'"{title}"\n' + " ".join(s.strip() for s in parts)
        for title, parts in zip(titles, parts_lists)
    ]


def _prepare_gold_context_dataset(hf_name, save_name, content_key, force=False):
    save_path = os.path.join(RAG_EVAL_DIR, save_name)
    if os.path.isdir(save_path) and not force:
        log("rag", f"  [skip] {save_path} already exists")
        return
    if force and os.path.isdir(save_path):
        shutil.rmtree(save_path)

    log("rag", f"  Loading {hf_name} from FlashRAG...")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", hf_name, split="dev")

    questions, contexts_list, answers_list = [], [], []
    for sample in ds:
        questions.append(sample["question"])
        answers_list.append(sample["golden_answers"])
        meta_ctx = (sample.get("metadata") or {}).get("context")
        if meta_ctx:
            contexts_list.append(_convert_metadata_context(meta_ctx, content_key))
        else:
            contexts_list.append([])

    out = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "golden_answers": answers_list,
    })
    out.save_to_disk(save_path)
    log("rag", f"  Saved {len(out)} samples → {save_path}")


def _prepare_retrieval_dataset(dataset_key, retriever, force=False):
    hf_name, save_name = RETRIEVAL_DATASETS[dataset_key]
    save_path = os.path.join(RAG_EVAL_DIR, save_name)
    if os.path.isdir(save_path) and not force:
        log("rag", f"  [skip] {save_path} already exists")
        return
    if force and os.path.isdir(save_path):
        shutil.rmtree(save_path)

    log("rag", f"  Loading {dataset_key} from FlashRAG...")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", hf_name, split="test")

    questions, contexts_list, answers_list = [], [], []
    for i, sample in enumerate(ds):
        questions.append(sample["question"])
        answers_list.append(sample["golden_answers"])
        if retriever is not None:
            try:
                contexts_list.append(retriever.retrieve(sample["question"], topk=10))
            except Exception as e:
                log("rag", f"  [WARN] BM25 retrieve failed at sample {i}: {e}. "
                    f"Falling back to closed-book for remaining samples.")
                retriever = None
                contexts_list.append([])
            if (i + 1) % 500 == 0 and retriever is not None:
                log("rag", f"    Retrieved {i + 1}/{len(ds)}...")
        else:
            contexts_list.append([])

    out = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "golden_answers": answers_list,
    })
    out.save_to_disk(save_path)
    mode = "with BM25 retrieval" if retriever is not None else "closed-book (no contexts)"
    log("rag", f"  Saved {len(out)} samples → {save_path} ({mode})")


def prepare_rag_eval_stage(with_retrieval, force=False, force_retrieval_only=False):
    log("rag", "stage=eval: preparing ../RAG_Eval/ datasets")
    os.makedirs(RAG_EVAL_DIR, exist_ok=True)

    if force_retrieval_only:
        for _, save_name in RETRIEVAL_DATASETS.values():
            p = os.path.join(RAG_EVAL_DIR, save_name)
            if os.path.isdir(p):
                log("rag", f"  [force-retrieval-only] removing {p}")
                shutil.rmtree(p)

    # Build retriever once if requested — graceful fallback to closed-book
    # --force-retrieval-only implies --with-retrieval (otherwise we'd just
    # delete the directories and rebuild them as closed-book, discarding contexts)
    retriever = None
    if with_retrieval or force_retrieval_only:
        try:
            retriever = BM25Retriever()
            retriever.build_index()
        except ImportError:
            log("rag", "  [WARN] rank_bm25 not installed; NQ/TQ/Bamboogle will be closed-book. "
                "Install with: pip install rank_bm25")
            retriever = None
        except Exception as e:
            log("rag", f"  [WARN] BM25 index build failed ({e}); falling back to closed-book")
            traceback.print_exc()
            retriever = None

    # Gold-context datasets
    log("rag", "  [HotpotQA] (gold contexts from FlashRAG)")
    _prepare_gold_context_dataset("hotpotqa", "HotpotQA_Eval", "sentences", force=force)
    log("rag", "  [2WikiMultihop] (gold contexts from FlashRAG)")
    _prepare_gold_context_dataset("2wikimultihopqa", "2Wiki_Eval", "content", force=force)

    # Retrieval-backed datasets
    ctx_note = "BM25 retrieval" if retriever is not None else "closed-book"
    for key in RETRIEVAL_DATASETS:
        log("rag", f"  [{key}] ({ctx_note})")
        _prepare_retrieval_dataset(key, retriever, force=force)


def prepare_rag(stage, with_retrieval=False, force=False, force_retrieval_only=False):
    if stage in ("train", "all"):
        prepare_rag_train(force=force)
    if stage in ("eval", "all"):
        prepare_rag_eval_stage(
            with_retrieval=with_retrieval,
            force=force,
            force_retrieval_only=force_retrieval_only,
        )


# ============================================================================
# Verification
# ============================================================================

def verify_math(stage="all"):
    ok = True
    splits = []
    if stage in ("train", "all"):
        splits.append("train")
    if stage in ("eval", "all"):
        splits.append("test")
    for split in splits:
        split_dir = os.path.join(MATH_DIR, split)
        if not os.path.isdir(split_dir):
            log("verify", f"  [MISSING] ../MATH/{split}/")
            ok = False
            continue
        subjects = [s for s in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, s))]
        count = sum(len(os.listdir(os.path.join(split_dir, s))) for s in subjects)
        log("verify", f"  [ok] ../MATH/{split}: {count} problems across {len(subjects)} subjects")
    return ok


def verify_mmlu_train():
    if os.path.isdir(MMLU_TRAIN_DIR):
        log("verify", f"  [ok] {MMLU_TRAIN_DIR}")
        return True
    log("verify", f"  [MISSING] {MMLU_TRAIN_DIR}")
    return False


def verify_rag_train():
    if os.path.isdir(RAG_TRAIN_DIR):
        log("verify", f"  [ok] {RAG_TRAIN_DIR}")
        return True
    log("verify", f"  [MISSING] {RAG_TRAIN_DIR}")
    return False


def verify_rag_eval_datasets():
    ok = True
    for name in ("NQ_Eval", "TQ_Eval", "2Wiki_Eval", "HotpotQA_Eval", "Bamboogle_Eval"):
        path = os.path.join(RAG_EVAL_DIR, name)
        if os.path.isdir(path):
            ds = Dataset.load_from_disk(path)
            has_ctx = any(len(c) > 0 for c in ds["contexts"][:10])
            ctx_status = "with contexts" if has_ctx else "closed-book"
            log("verify", f"  [ok] {name}: {len(ds)} samples ({ctx_status})")
        else:
            log("verify", f"  [MISSING] {name}")
            ok = False
    return ok


# ============================================================================
# Main
# ============================================================================

ALL_TASKS = ("gsm8k", "math", "mmlu", "rag")


def _parse_tasks(tasks_arg):
    """Parse the `--tasks` flag into a validated task list."""
    if tasks_arg == "all":
        return list(ALL_TASKS)
    out = []
    for t in tasks_arg.split(","):
        t = t.strip()
        if not t:
            continue
        if t not in ALL_TASKS:
            raise SystemExit(f"Unknown task: {t!r} (valid: {','.join(ALL_TASKS)} or 'all')")
        out.append(t)
    return out


def main():
    """CLI entrypoint for preparing repository datasets and verifying outputs."""
    parser = argparse.ArgumentParser(
        description="Unified GRPO/TGRPO/HRPO/THRPO data preparation (train + eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help="Comma-separated: gsm8k,math,mmlu,rag or 'all' (default: all)",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("train", "eval", "all"),
        help="Which stage(s) to prepare (default: all)",
    )
    parser.add_argument(
        "--with-retrieval",
        action="store_true",
        help="Build BM25 index for RAG NQ/TQ/Bamboogle eval "
             "(silently ignored for non-rag tasks; requires rank_bm25)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-prep even if outputs exist",
    )
    parser.add_argument(
        "--force-retrieval-only",
        action="store_true",
        help="Re-prep only NQ/TQ/Bamboogle RAG eval (convenience flag)",
    )
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)

    log_section(f"prepare_data.py — tasks={','.join(tasks)} stage={args.stage}")

    failures = []

    dispatch = {
        "gsm8k": lambda: prepare_gsm8k(args.stage, force=args.force),
        "math": lambda: prepare_math(args.stage, force=args.force),
        "mmlu": lambda: prepare_mmlu(args.stage, force=args.force),
        "rag": lambda: prepare_rag(
            args.stage,
            with_retrieval=args.with_retrieval,
            force=args.force,
            force_retrieval_only=args.force_retrieval_only,
        ),
    }

    for task in tasks:
        log_section(f"Preparing: {task}")
        try:
            dispatch[task]()
        except Exception as e:
            log(task, f"[ERROR] {e}")
            traceback.print_exc()
            failures.append(task)

    # Verification pass (scoped to selected tasks/stages)
    log_section("Verification")
    if "math" in tasks:
        verify_math(stage=args.stage)
    if "mmlu" in tasks and args.stage in ("train", "all"):
        verify_mmlu_train()
    if "rag" in tasks:
        if args.stage in ("train", "all"):
            verify_rag_train()
        if args.stage in ("eval", "all"):
            verify_rag_eval_datasets()

    if failures:
        log("MAIN", f"Completed with failures in: {', '.join(failures)}")
        sys.exit(1)
    log("MAIN", "All requested tasks prepared successfully.")


if __name__ == "__main__":
    main()
