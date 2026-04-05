#!/usr/bin/env python
"""
prepare_eval_data.py — Prepare all evaluation datasets for HRPO reproduction.

Prepares:
  1. RAG eval datasets (../RAG_Eval/): NQ, TriviaQA, 2Wiki, HotpotQA, Bamboogle
  2. MATH test data at ../MATH/test/ (auto-downloads with --download-math)
  3. GSM8K and MMLU-STEM are auto-downloaded by eval scripts (no prep needed)

Usage:
  python prepare_eval_data.py                    # Prepare all eval datasets
  python prepare_eval_data.py --only-rag         # Only prepare RAG eval datasets
  python prepare_eval_data.py --with-retrieval   # Also build retrieval contexts for NQ/TQ/Bamboogle
  python prepare_eval_data.py --download-math    # Auto-download MATH dataset if missing

RAG eval datasets format:
  Each saved dataset has columns: question (str), contexts (list[str]), golden_answers (list[str])
  - HotpotQA, 2Wiki: gold contexts from FlashRAG metadata (~10 paragraphs each)
  - NQ, TriviaQA, Bamboogle: requires --with-retrieval for context passages,
    otherwise saved with empty contexts (closed-book evaluation)

Note on retrieval method:
  The HRPO paper uses E5 embedding model (intfloat/e5-base-v2) with approximate nearest
  neighbor (ANN) search over the full English Wikipedia 2020 dump to retrieve top-3
  documents per query. The --with-retrieval flag here uses a lightweight BM25 approximation
  over HotpotQA+2Wiki paragraphs (max 50k) for convenience. For exact paper reproduction,
  E5+Wikipedia retrieval should be used instead.
"""

import os
import argparse
from datasets import load_dataset, Dataset


RAG_EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "RAG_Eval")
MATH_DIR = os.path.join(os.path.dirname(__file__), "..", "MATH")


# ============================================================================
# Context conversion helpers
# ============================================================================

def convert_metadata_context(metadata_context, content_key):
    """Convert metadata.context (title + sentences/content) to list of passage strings.

    Args:
        metadata_context: dict with 'title' and content_key lists
        content_key: 'sentences' for HotpotQA, 'content' for 2Wiki
    """
    titles = metadata_context["title"]
    content_lists = metadata_context[content_key]
    return [f'"{title}"\n' + " ".join(s.strip() for s in parts)
            for title, parts in zip(titles, content_lists)]


# ============================================================================
# BM25 Retriever (lightweight, used for --with-retrieval)
# ============================================================================

class BM25Retriever:
    """Simple BM25 retriever using rank_bm25 over HotpotQA+2Wiki paragraphs."""

    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank_bm25 is required for --with-retrieval. "
                "Install with: pip install rank_bm25"
            )
        self.BM25Okapi = BM25Okapi
        self.corpus = []
        self.corpus_texts = []
        self.index = None

    def build_index(self, max_paragraphs=50000):
        """Build BM25 index from HotpotQA Wikipedia paragraphs (sampled to fit in memory)."""
        print(f"Building BM25 index (max {max_paragraphs} paragraphs)...")

        seen_titles = set()

        # Load HotpotQA paragraphs (unique by title to reduce duplicates)
        hotpot = load_dataset(
            "RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="dev"
        )
        for sample in hotpot:
            if len(self.corpus) >= max_paragraphs:
                break
            if "metadata" in sample and sample["metadata"]:
                ctx = sample["metadata"].get("context", {})
                titles = ctx.get("title", [])
                sents = ctx.get("sentences", [])
                for title, sent_list in zip(titles, sents):
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    passage = f'"{title}"\n' + " ".join(s.strip() for s in sent_list)
                    self.corpus.append(passage)
                    self.corpus_texts.append(
                        title.lower() + " " + " ".join(s.strip().lower() for s in sent_list)
                    )
                    if len(self.corpus) >= max_paragraphs:
                        break

        # Add 2Wiki paragraphs if we still have room
        if len(self.corpus) < max_paragraphs:
            wiki2 = load_dataset(
                "RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split="dev"
            )
            for sample in wiki2:
                if len(self.corpus) >= max_paragraphs:
                    break
                if "metadata" in sample and sample["metadata"]:
                    ctx = sample["metadata"].get("context", {})
                    titles = ctx.get("title", [])
                    contents = ctx.get("content", [])
                    for title, content_list in zip(titles, contents):
                        if title in seen_titles:
                            continue
                        seen_titles.add(title)
                        passage = f'"{title}"\n' + " ".join(s.strip() for s in content_list)
                        self.corpus.append(passage)
                        self.corpus_texts.append(
                            title.lower() + " " + " ".join(
                                s.strip().lower() for s in content_list
                            )
                        )
                        if len(self.corpus) >= max_paragraphs:
                            break

        print(f"  Corpus size: {len(self.corpus)} unique paragraphs")
        tokenized = [text.split() for text in self.corpus_texts]
        self.corpus_texts = None  # Free memory before building index
        self.index = self.BM25Okapi(tokenized)
        print("  BM25 index built.")

    def retrieve(self, query, topk=10):
        """Retrieve top-k passages for a query."""
        if self.index is None:
            self.build_index()
        scores = self.index.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [self.corpus[i] for i in top_indices]


# ============================================================================
# Dataset preparation functions
# ============================================================================

def prepare_hotpotqa(output_dir):
    """Prepare HotpotQA evaluation dataset with gold contexts."""
    save_path = os.path.join(output_dir, "HotpotQA_Eval")
    if os.path.exists(save_path):
        print(f"  [skip] {save_path} already exists")
        return

    print("  Loading HotpotQA from FlashRAG...")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="dev")

    questions, contexts_list, answers_list = [], [], []
    for sample in ds:
        questions.append(sample["question"])
        answers_list.append(sample["golden_answers"])
        if "metadata" in sample and sample["metadata"] and "context" in sample["metadata"]:
            contexts_list.append(convert_metadata_context(sample["metadata"]["context"], "sentences"))
        else:
            contexts_list.append([])

    out = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "golden_answers": answers_list,
    })
    out.save_to_disk(save_path)
    print(f"  Saved {len(out)} samples to {save_path}")


def prepare_2wiki(output_dir):
    """Prepare 2WikiMultihop evaluation dataset with gold contexts."""
    save_path = os.path.join(output_dir, "2Wiki_Eval")
    if os.path.exists(save_path):
        print(f"  [skip] {save_path} already exists")
        return

    print("  Loading 2WikiMultihopQA from FlashRAG...")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split="dev")

    questions, contexts_list, answers_list = [], [], []
    for sample in ds:
        questions.append(sample["question"])
        answers_list.append(sample["golden_answers"])
        if "metadata" in sample and sample["metadata"] and "context" in sample["metadata"]:
            contexts_list.append(convert_metadata_context(sample["metadata"]["context"], "content"))
        else:
            contexts_list.append([])

    out = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "golden_answers": answers_list,
    })
    out.save_to_disk(save_path)
    print(f"  Saved {len(out)} samples to {save_path}")


RETRIEVAL_DATASETS = {
    'nq': ('nq', 'NQ_Eval'),
    'triviaqa': ('triviaqa', 'TQ_Eval'),
    'bamboogle': ('bamboogle', 'Bamboogle_Eval'),
}


def prepare_retrieval_dataset(dataset_key, output_dir, retriever=None):
    """Prepare a FlashRAG retrieval dataset (NQ, TriviaQA, or Bamboogle)."""
    hf_name, save_name = RETRIEVAL_DATASETS[dataset_key]
    save_path = os.path.join(output_dir, save_name)
    if os.path.exists(save_path):
        print(f"  [skip] {save_path} already exists")
        return

    print(f"  Loading {dataset_key} from FlashRAG...")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", hf_name, split="test")

    questions, contexts_list, answers_list = [], [], []
    for i, sample in enumerate(ds):
        questions.append(sample["question"])
        answers_list.append(sample["golden_answers"])
        if retriever:
            contexts_list.append(retriever.retrieve(sample["question"], topk=10))
            if (i + 1) % 500 == 0:
                print(f"    Retrieved {i+1}/{len(ds)}...")
        else:
            contexts_list.append([])

    out = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "golden_answers": answers_list,
    })
    out.save_to_disk(save_path)
    mode = "with BM25 retrieval" if retriever else "without contexts (closed-book)"
    print(f"  Saved {len(out)} samples to {save_path} ({mode})")


# ============================================================================
# Verification helpers
# ============================================================================

def download_math():
    """Download MATH dataset from HuggingFace and save to ../MATH/ in expected folder structure."""
    if os.path.isdir(os.path.join(MATH_DIR, "test")):
        print("  [skip] MATH dataset already exists")
        return True

    print("  Downloading MATH dataset from EleutherAI/hendrycks_math...")
    import json as _json

    os.makedirs(MATH_DIR, exist_ok=True)

    # EleutherAI/hendrycks_math has per-subject configs
    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    for split in ["train", "test"]:
        for subject in subjects:
            out_dir = os.path.join(MATH_DIR, split, subject)
            os.makedirs(out_dir, exist_ok=True)

            try:
                ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
            except Exception as e:
                print(f"  [WARN] Failed to load {subject}/{split}: {e}")
                continue

            for idx, sample in enumerate(ds):
                filepath = os.path.join(out_dir, f"{idx}.json")
                with open(filepath, "w") as f:
                    _json.dump({
                        "problem": sample["problem"],
                        "solution": sample["solution"],
                    }, f)

        total = sum(
            len(os.listdir(os.path.join(MATH_DIR, split, s)))
            for s in subjects if os.path.isdir(os.path.join(MATH_DIR, split, s))
        )
        print(f"  Downloaded MATH {split}: {total} problems")

    return True


def verify_math():
    """Check that MATH test data exists."""
    test_dir = os.path.join(MATH_DIR, "test")
    if not os.path.isdir(test_dir):
        print(f"  [WARN] MATH test data not found at {test_dir}")
        print("         Run with --download-math to auto-download, or manually place data")
        return False
    subjects = os.listdir(test_dir)
    count = sum(
        len(os.listdir(os.path.join(test_dir, s)))
        for s in subjects if os.path.isdir(os.path.join(test_dir, s))
    )
    print(f"  [ok] MATH test: {count} problems across {len(subjects)} subjects")
    return True


def verify_rag_eval(output_dir):
    """Verify all RAG eval datasets exist and show stats."""
    all_ok = True
    for name in ["NQ_Eval", "TQ_Eval", "2Wiki_Eval", "HotpotQA_Eval", "Bamboogle_Eval"]:
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            ds = Dataset.load_from_disk(path)
            has_ctx = any(len(c) > 0 for c in ds["contexts"][:10])
            ctx_status = "with contexts" if has_ctx else "NO contexts (closed-book)"
            print(f"  [ok] {name}: {len(ds)} samples ({ctx_status})")
        else:
            print(f"  [MISSING] {name}")
            all_ok = False
    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare HRPO evaluation datasets")
    parser.add_argument("--only-rag", action="store_true",
                        help="Only prepare RAG eval datasets")
    parser.add_argument("--with-retrieval", action="store_true",
                        help="Use BM25 retrieval for NQ/TriviaQA/Bamboogle contexts "
                             "(requires: pip install rank_bm25)")
    parser.add_argument("--force", action="store_true",
                        help="Re-prepare even if datasets already exist")
    parser.add_argument("--force-retrieval-only", action="store_true",
                        help="Only re-prepare NQ/TQ/Bamboogle (the ones needing retrieval)")
    parser.add_argument("--download-math", action="store_true",
                        help="Auto-download MATH dataset from EleutherAI/hendrycks_math if missing")
    args = parser.parse_args()

    print("=" * 60)
    print("HRPO Evaluation Data Preparation")
    print("=" * 60)

    # Force mode: remove existing RAG eval datasets
    if args.force and os.path.isdir(RAG_EVAL_DIR):
        import shutil
        print(f"[force] Removing existing {RAG_EVAL_DIR}")
        shutil.rmtree(RAG_EVAL_DIR)
    elif args.force_retrieval_only:
        import shutil
        for name in ["NQ_Eval", "TQ_Eval", "Bamboogle_Eval"]:
            path = os.path.join(RAG_EVAL_DIR, name)
            if os.path.isdir(path):
                print(f"[force-retrieval-only] Removing {path}")
                shutil.rmtree(path)

    # 1. Verify non-RAG eval data
    if not args.only_rag:
        print("\n--- Checking non-RAG eval datasets ---")
        print("[GSM8K] Auto-downloads from HuggingFace (openai/gsm8k) — no prep needed")
        print("[MMLU-STEM] Auto-downloads from HuggingFace (TIGER-Lab/MMLU-STEM) — no prep needed")
        print("[ARC-C] Auto-downloads from HuggingFace (allenai/ai2_arc) — no prep needed")
        print("[MATH]")
        if args.download_math:
            download_math()
        verify_math()

    # 2. Prepare RAG eval datasets
    print("\n--- Preparing RAG eval datasets ---")
    os.makedirs(RAG_EVAL_DIR, exist_ok=True)

    # Build retriever if requested
    retriever = None
    if args.with_retrieval:
        print("\n[Retrieval] Building BM25 index...")
        retriever = BM25Retriever()
        retriever.build_index()

    # Prepare datasets with gold contexts first (used as retrieval corpus)
    print("\n[HotpotQA] (gold contexts from FlashRAG)")
    prepare_hotpotqa(RAG_EVAL_DIR)

    print("\n[2WikiMultihop] (gold contexts from FlashRAG)")
    prepare_2wiki(RAG_EVAL_DIR)

    # Prepare datasets that need retrieval
    ctx_note = "BM25 retrieval" if retriever else "no contexts"
    for key in RETRIEVAL_DATASETS:
        print(f"\n[{key}] ({ctx_note})")
        prepare_retrieval_dataset(key, RAG_EVAL_DIR, retriever)

    # 3. Final verification
    print("\n--- Verification ---")
    verify_rag_eval(RAG_EVAL_DIR)

    if not retriever:
        print("\n[NOTE] NQ, TriviaQA, and Bamboogle were saved WITHOUT retrieval contexts.")
        print("       For full RAG evaluation, re-run with: python prepare_eval_data.py --with-retrieval")
        print("       (requires: pip install rank_bm25)")

    print("\nDone!")


if __name__ == "__main__":
    main()
