import os
import ast
import json
import argparse
import warnings
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from tqdm import tqdm

from tools.utils import APIClient
from tools.prompts import final_response_cv_template, passage_selection_template

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class Explorer:
    """
    Prune-and-Grow over the memory hierarchy.

    Steps:
      1) Cosine top-k over all-level embeddings → initial candidates S.
      2) LLM selects useful subset X from S.
      3) Expand X with same-level neighbors + lower-level community members.
      4) Re-select on expanded pool; repeat until stable or reaching max turns.
      5) Merge contiguous level-0 chunks; answer with LLM.
    """

    def __init__(
        self,
        dataset: str,
        book_title: str,
        api_key_path: str = "openai_key.txt",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        text_field: str = "gist",
        top_k: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.dataset = dataset
        self.book_title = book_title
        self.text_field = text_field
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.graph_path = f"./super_graphs/{dataset}/{book_title}/{book_title}_graph_level_all.gexf"
        self.embedding_path = f"./super_embeddings/{dataset}/{book_title}/{book_title}_embedding_level_all.npy"

        self.memory_graph = self._load_memory_graph()
        self.memory_embedding = np.load(self.embedding_path).astype(np.float32)

        self.client = APIClient("openai", api_key_path, model, embedding_model)

        self.num_level0 = sum(1 for _, d in self.memory_graph.nodes(data=True) if d.get("level", 0) == 0)

    def _load_memory_graph(self) -> nx.Graph:
        """Load the GEXF file."""
        G_loaded = nx.read_gexf(self.graph_path)
        for node, data in G_loaded.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                    G_loaded.nodes[node][key] = ast.literal_eval(value)
        mapping = {node: int(node) for node in G_loaded.nodes()}
        G_loaded = nx.relabel_nodes(G_loaded, mapping)
        return G_loaded

    def _embed_claim(self, claim: str) -> np.ndarray:
        emb = self.client.obtain_embedding(claim)
        return np.array(emb, dtype=np.float32)

    def _llm_selection_once(self, question: str, candidates: List[Tuple[int, str]]) -> Optional[List[int]]:
        """
        Ask LLM to select useful passages (1-based indices). Minimal parsing.
        Returns 0-based indices list on success; None on parse failure.
        """
        if not candidates:
            return []

        txt = "\n".join(f"Passage {i+1}: {c[1]}" for i, c in enumerate(candidates))
        prompt = passage_selection_template.format(
            input_question=question + " Is this statement true or false?",
            passages=txt
        )
        resp = self.client.obtain_response(prompt, max_tokens=self.max_tokens, temperature=self.temperature).strip()

        try:
            idx_list = ast.literal_eval(resp)  # expect a python-like list
            out = set()
            for i in idx_list:
                j = int(i) - 1
                if 0 <= j < len(candidates):
                    out.add(j)
            return sorted(list(out))
        except Exception:
            return None

    def run(self,
            claim: str,
            max_exploration_turns: int = 5,
            tolerance: int = 3) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Returns:
          - prediction (str)
          - used_passages (List[Tuple[int, str]])
        """
        q_emb = self._embed_claim(claim)

        # initial top-k
        top_nodes = self._initial_topk(q_emb)
        node_text = [(nid, self._node_text(nid)) for nid in top_nodes]

        init_retrieved = node_text.copy()
        all_retrieved: List[Tuple[int, str]] = []
        prev_set: set[int] = set()

        turn = 0
        local_tol = tolerance
        while turn < max_exploration_turns:
            curr_set = set(n for n, _ in node_text)
            print(f"[Turn {turn+1}] candidates: {len(curr_set)} | nodes: {sorted(list(curr_set))[:30]}{' ...' if len(curr_set)>30 else ''}")

            if curr_set == prev_set:
                print("[EarlyStop] set stabilized.")
                break

            chosen_local_idx = self._llm_selection_once(claim, node_text)
            if chosen_local_idx is None:
                print("[Select] parse failed, retrying...")
                if local_tol > 0:
                    local_tol -= 1
                    continue
                else:
                    print("[Select] tolerance exhausted, stop.")
                    break

            prev_set = set(curr_set)

            if len(chosen_local_idx) == 0:
                print("[Select] empty selection.")
                turn += 1
                continue

            chosen_nodes = [node_text[i][0] for i in chosen_local_idx]

            expanded = set()
            for nid in chosen_nodes:
                all_retrieved.append((nid, self._node_text(nid)))
                neighbors = list(self.memory_graph.neighbors(nid))
                comm = self.memory_graph.nodes[nid].get("community", [])
                expanded.update(neighbors)
                expanded.update(comm)
                expanded.add(nid)

            combined = {nid: self._node_text(nid) for nid in expanded}
            node_text = list(combined.items())
            turn += 1

        # Final evidence
        combined_evidence = {nid: txt for nid, txt in all_retrieved}
        if not combined_evidence:
            combined_evidence = {nid: txt for nid, txt in init_retrieved}

        # Merge contiguous level-0 chunks
        merged_passages = self._merge_level0_chunks(list(combined_evidence.items()))

        # Answer
        context_str = "\n".join(f"Passage {i+1}: {txt}" for i, (_, txt) in enumerate(merged_passages))
        prompt = final_response_cv_template.format(
            question=claim,
            passages=context_str
        )
        prediction = self.client.obtain_response(prompt, max_tokens=self.max_tokens, temperature=self.temperature).strip()
        return prediction, merged_passages

    def _initial_topk(self, q_emb: np.ndarray) -> List[int]:
        scores = cosine_similarity(q_emb[np.newaxis, :], self.memory_embedding)[0]
        idx = np.argsort(-scores)[: self.top_k]
        print(f"[Init] top-{self.top_k} idx: {idx.tolist()}")
        return idx.tolist()

    def _node_text(self, node_id: int) -> str:
        n = int(node_id)
        txt = self.memory_graph.nodes[n].get(self.text_field, "")
        return txt.strip() if isinstance(txt, str) else ""

    def _merge_level0_chunks(self, pairs: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        if not pairs:
            return []

        pairs = sorted(pairs, key=lambda x: x[0])
        result: List[Tuple[int, str]] = []

        def is_level0(nid: int) -> bool:
            return self.memory_graph.nodes[nid].get("level", 0) == 0

        if not is_level0(pairs[0][0]):
            result.extend(pairs)
            return result

        run_ids = [pairs[0][0]]
        run_txt = [pairs[0][1]]

        for i in range(1, len(pairs)):
            nid, txt = pairs[i]
            prev_id, _ = pairs[i - 1]
            if is_level0(nid) and is_level0(prev_id) and (nid == prev_id + 1):
                run_ids.append(nid)
                run_txt.append(txt)
            else:
                result.append((run_ids[-1], " ".join(run_txt)))
                if not is_level0(nid):
                    result.extend(pairs[i:])
                    return result
                run_ids = [nid]
                run_txt = [txt]

        result.append((run_ids[-1], " ".join(run_txt)))
        return result


def parse_args():
    p = argparse.ArgumentParser(description="Claim Verification with CAM")
    p.add_argument("--dataset", type=str, default="FABLES")
    p.add_argument("--claims_dir", type=str, default=None, help="Default: ./data/<dataset>/claims/")

    p.add_argument("--api_key_path", type=str, default="openai_key.txt")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    p.add_argument("--text_field", type=str, default="text", help="Node text attribute to read (e.g., 'gist' or 'raw_text').")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--tolerance", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=500)

    p.add_argument("--save_dir", type=str, default="./output")
    p.add_argument("--save_name", type=str, default="CAM_CV_Output.json")
    p.add_argument("--save_evidence", action="store_true", help="Save retrieved evidence passages")
    return p.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset
    claims_path = args.claims_dir or f'./data/{dataset}/claims/'
    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.save_name)

    prediction_json = []
    total_num = 0
    correct_num = 0

    # Per-label counts
    tp_true = fp_true = fn_true = 0
    tp_false = fp_false = fn_false = 0

    files = [f for f in os.listdir(claims_path) if f.endswith(".json")]
    files.sort()

    for file_name in files:
        book_title = os.path.splitext(file_name)[0]
        print(f"\n=== Verifying Claims in {book_title} ===")

        gexf = f"./super_graphs/{dataset}/{book_title}/{book_title}_graph_level_all.gexf"
        npy = f"./super_embeddings/{dataset}/{book_title}/{book_title}_embedding_level_all.npy"
        if not (os.path.exists(gexf) and os.path.exists(npy)):
            print(f"[Skip] No composed memory found for {book_title}.")
            continue

        recall = Explorer(
            dataset=dataset,
            book_title=book_title,
            api_key_path=args.api_key_path,
            model=args.model,
            embedding_model=args.embedding_model,
            text_field=args.text_field,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        with open(os.path.join(claims_path, file_name), "r") as f:
            claim_list = json.load(f)

        total_num += len(claim_list)
        local_correct = 0

        for cd in tqdm(claim_list, leave=False):
            claim = cd["claim"]
            label = cd["label"]  # "True" or "False"

            prediction, used_passages = recall.run(
                claim=claim,
                max_exploration_turns=args.max_turns,
                tolerance=args.tolerance,
            )

            print(f"Claim: {claim}")
            print(f"Prediction: {prediction}")
            print(f"Gold: {label}")

            # Normalize
            pred_norm = str(prediction).strip().lower()
            gold_norm = str(label).strip().lower()

            is_correct = (pred_norm == gold_norm)
            local_correct += 1 if is_correct else 0

            # Update per-label counts
            if gold_norm == "yes":
                if pred_norm == "yes":
                    tp_true += 1
                else:
                    fn_true += 1
                    fp_false += 1
            else:  # gold is "false"
                if pred_norm == "no":
                    tp_false += 1
                else:
                    fn_false += 1
                    fp_true += 1

            rec = {
                "Book": book_title,
                "Claim": claim,
                "Gold": label,
                "Prediction": prediction,
                "Correct": is_correct,
            }
            if args.save_evidence:
                rec["UsedPassages"] = [{"node_id": nid, "text": txt} for nid, txt in used_passages]
            prediction_json.append(rec)

        acc = local_correct / max(1, len(claim_list))
        print(f"[{book_title}] Accuracy: {acc:.4f}")

        correct_num += local_correct
        print("=" * 25)

    # Summary
    if total_num == 0:
        print("[Warn] No claims processed.")
    else:
        overall_acc = correct_num / total_num
        print(f"\n== Overall Accuracy: {overall_acc:.4f} ==")

        prec_true, rec_true, f1_true = calculate_f1(tp_true, fp_true, fn_true)
        prec_false, rec_false, f1_false = calculate_f1(tp_false, fp_false, fn_false)

        print(f"True  - Precision: {prec_true:.4f}, Recall: {rec_true:.4f}, F1: {f1_true:.4f}")
        print(f"False - Precision: {prec_false:.4f}, Recall: {rec_false:.4f}, F1: {f1_false:.4f}")

    # Save
    with open(out_path, "w") as f:
        json.dump(prediction_json, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()