import os
import ast
import json
import argparse
import warnings
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

from tools.utils import APIClient
from tools.prompts import (
    final_response_mc_template,
    final_response_ge_template,
    passage_selection_template,
)

warnings.filterwarnings("ignore", category=UserWarning)


def llm_judge(client: APIClient,
              question: str,
              prediction: str,
              answer: str,
              max_tokens: int = 500,
              temperature: float = 0.0) -> float:
    """
    Use an LLM to judge if `prediction` semantically matches the gold `answer`.
    Returns 1.0 for "Yes", 0.0 otherwise.
    """
    from tools.prompts import judge_template
    prompt = judge_template.format(question=question, prediction=prediction, answer=answer)
    resp = client.obtain_response(prompt, max_tokens=max_tokens, temperature=temperature).strip()
    return 1.0 if resp.lower() == "yes" else 0.0


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

    def _embed_question(self, question: str) -> np.ndarray:
        emb = self.client.obtain_embedding(question)
        return np.array(emb, dtype=np.float32)

    def _llm_selection(self, question: str, candidates: List[Tuple[int, str]]) -> Optional[List[int]]:
        if not candidates:
            return []

        txt = "\n".join(f"Passage {i+1}: {c[1]}" for i, c in enumerate(candidates))
        prompt = passage_selection_template.format(input_question=question, passages=txt)
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
            question: str,
            mode: str = "MC",
            options: Optional[List[str]] = None,
            max_exploration_turns: int = 5,
            tolerance: int = 2) -> Tuple[str, List[int], List[Tuple[int, str]]]:
        """
        Returns:
          - prediction (str)
          - used_passages (List[Tuple[int, str]])
        """
        q_emb = self._embed_question(question)

        # Initial top-k
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

            chosen_local_idx = self._llm_selection(question, node_text)
            if chosen_local_idx is None:
                print("[Select] parse failed, retrying...")
                if local_tol > 0:
                    local_tol -= 1
                    continue
                else:
                    print("[Select] tolerance exhausted, stop.")
                    break

            prev_set = set(curr_set)  # Only update once we have a valid selection

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
        if mode.upper() == "MC":
            prompt = final_response_mc_template.format(
                question=question,
                options=options,
                passages=context_str
            )
        else:
            prompt = final_response_ge_template.format(
                question=question,
                passages=context_str
            )
        prediction = self.client.obtain_response(prompt, max_tokens=self.max_tokens, temperature=self.temperature).strip()
        return prediction, merged_passages

    def _initial_topk(self, q_emb: np.ndarray) -> List[int]:
        scores = cosine_similarity(q_emb[np.newaxis, :], self.memory_embedding)[0]
        # scores = np.maximum(0, scores)
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
    p = argparse.ArgumentParser(description="QA with CAM")
    p.add_argument("--dataset", type=str, default="NovelQA")
    p.add_argument("--question_dir", type=str, default=None, help="./data/<dataset>/questions/")
    p.add_argument("--mode", type=str, choices=["MC", "GE"], default="MC")

    p.add_argument("--api_key_path", type=str, default="openai_key.txt")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    p.add_argument("--text_field", type=str, default="text", help="Node text attribute to read.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--tolerance", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=512)

    p.add_argument("--save_dir", type=str, default="./output")
    p.add_argument("--save_name", type=str, default=None, help="Override default output filename")
    p.add_argument("--save_evidence", action="store_true", help="Also store retrieved passage ids/texts")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    return p.parse_args()


def _process_file(file_name, cfg):
    # Result container
    result = {
        'prediction_json': [],
        'total_num': 0,
        'correct_num': 0,
        'all_r1': 0.0,
        'all_r2': 0.0,
        'all_rl': 0.0,
    }

    book_title = os.path.splitext(file_name)[0]
    print(f"\n=== Answering Questions in {book_title} ===")

    # Ensure reading memory exists
    dataset = cfg["dataset"]
    gexf = f"./super_graphs/{dataset}/{book_title}/{book_title}_graph_level_all.gexf"
    npy = f"./super_embeddings/{dataset}/{book_title}/{book_title}_embedding_level_all.npy"
    if not (os.path.exists(gexf) and os.path.exists(npy)):
        print(f"[Skip] No reading memory found for {book_title}.")
        return result

    api_key_path = cfg["api_key_path"]
    model = cfg["model"]
    embedding_model = cfg["embedding_model"]
    text_field = cfg["text_field"]
    top_k = cfg["top_k"]
    temperature = cfg["temperature"]
    max_tokens = cfg["max_tokens"]
    recall = Explorer(
        dataset=dataset,
        book_title=book_title,
        api_key_path=api_key_path,
        model=model,
        embedding_model=embedding_model,
        text_field=text_field,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    question_path = cfg["question_path"]
    with open(os.path.join(question_path, file_name), "r") as f:
        question_list = json.load(f)

    result['total_num'] += len(question_list)
    local_correct = 0
    mode = cfg["mode"]
    if mode == "GE":
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1 = r2 = rl = 0.0

    max_turns = cfg["max_turns"]
    tolerance = cfg["tolerance"]
    save_evidence = cfg["save_evidence"]
    for qd in tqdm(question_list, leave=False):
        qid = qd.get("QID")
        question = qd["Question"]
        answer = qd.get("Answer")
        options = qd.get("Options")
        gold = qd.get("Gold")
        aspect = qd.get("Aspect")
        complexity = qd.get("Complexity")

        prediction, used_passages = recall.run(
            question=question,
            mode=mode,
            options=options,
            max_exploration_turns=max_turns,
            tolerance=tolerance,
        )

        print(f"Question: {question}")
        print(f"Prediction: {prediction}")

        if mode == "MC":
            is_correct = (str(prediction).strip().lower() == str(gold).strip().lower())
            local_correct += 1 if is_correct else 0
            rec = {
                "QID": qid,
                "Aspect": aspect,
                "Complexity": complexity,
                "Question": question,
                "Options": options,
                "Gold": gold,
                "Prediction": prediction,
                "Correct": is_correct,
            }
            print(f"Gold: {gold}")
            print("Result: ", "Correct" if is_correct else "Incorrect")
        else:
            scores = scorer.score(answer, prediction)
            r1 += scores["rouge1"].fmeasure
            r2 += scores["rouge2"].fmeasure
            rl += scores["rougeL"].fmeasure

            score_llm = llm_judge(
                client=recall.client,
                question=question,
                prediction=prediction,
                answer=answer,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            local_correct += score_llm

            rec = {
                "QID": qid,
                "Aspect": aspect,
                "Complexity": complexity,
                "Question": question,
                "Answer": answer,
                "Prediction": prediction,
                "LLM_Judge": score_llm,
                "ROUGE-1": scores["rouge1"].fmeasure,
                "ROUGE-2": scores["rouge2"].fmeasure,
                "ROUGE-L": scores["rougeL"].fmeasure,
            }
            print("Answer: ", answer)
            print("LLM Judge", score_llm)
            print("ROUGE-1 Score: ", scores['rouge1'].fmeasure)
            print("ROUGE-2 Score: ", scores['rouge2'].fmeasure)
            print("ROUGE-L Score: ", scores['rougeL'].fmeasure)

        if save_evidence:
            rec["UsedPassages"] = [{"node_id": nid, "text": txt} for nid, txt in used_passages]

        result['prediction_json'].append(rec)

    if mode == "MC":
        acc = local_correct / max(1, len(question_list))
        result['correct_num'] += local_correct
        print(f"[{book_title}] QA accuracy (MC): {acc:.4f}")
    else:
        n = max(1, len(question_list))
        print(f"[{book_title}] ROUGE-1: {r1/n:.4f} | ROUGE-2: {r2/n:.4f} | ROUGE-L: {rl/n:.4f} | LLM_Judge_Acc: {local_correct/n:.4f}")
        result['correct_num'] += local_correct
        result['all_r1'] += r1
        result['all_r2'] += r2
        result['all_rl'] += rl
    print("=" * 25)

    return result


def main():
    args = parse_args()

    dataset = args.dataset
    question_path = args.question_dir or f'./data/{dataset}/questions/'
    mode = args.mode.upper()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name:
        out_path = os.path.join(args.save_dir, args.save_name)
    else:
        out_suffix = "MC" if mode == "MC" else "GE"
        out_path = os.path.join(args.save_dir, f"CAM_QA_Output ({out_suffix} setting).json")

    prediction_json = []
    total_num = 0
    correct_num = 0

    if mode == "GE":
        all_r1 = all_r2 = all_rl = 0.0

    files = [f for f in os.listdir(question_path) if f.endswith(".json")]
    files.sort()
    cfg = {
        "dataset": dataset,
        "question_path": question_path,
        "mode": mode,
        "api_key_path": args.api_key_path,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "text_field": args.text_field,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_turns": args.max_turns,
        "tolerance": args.tolerance,
        "save_evidence": args.save_evidence,
    }

    if args.workers <= 1 or len(files) <= 1:
        for file_name in files:
            res = _process_file(file_name, cfg)

            prediction_json.extend(res['prediction_json'])
            total_num += res['total_num']
            correct_num += res['correct_num']

            if mode == "GE":
                all_r1 += res['all_r1']
                all_r2 += res['all_r2']
                all_rl += res['all_rl']
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_file, file_name, cfg): file_name for file_name in files}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Files"):
                res = fut.result()

                prediction_json.extend(res['prediction_json'])
                total_num += res['total_num']
                correct_num += res['correct_num']

                if mode == "GE":
                    all_r1 += res['all_r1']
                    all_r2 += res['all_r2']
                    all_rl += res['all_rl']

    # Summary
    if total_num == 0:
        print("[Warn] No questions processed.")
    else:
        if mode == "MC":
            print(f"\n== Final MC accuracy: {correct_num / total_num:.4f} ==")
        else:
            print(f"\n== Final GE =="
                  f"\nROUGE-1: {all_r1/total_num:.4f}"
                  f"\nROUGE-2: {all_r2/total_num:.4f}"
                  f"\nROUGE-L: {all_rl/total_num:.4f}"
                  f"\nLLM_Judge_Acc: {correct_num/total_num:.4f}")

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(prediction_json, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
