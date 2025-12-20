import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert corpus.json to per-document .txt files for chunks.py")
    parser.add_argument("--input", "-i", type=str, default="data/MultiHopRAG/MultiHopRAG.json", help="Path to json file")
    parser.add_argument("--outdir", "-o", type=str, default="data/MultiHopRAG/questions", help="Output directory for json files")
    args = parser.parse_args()

    # Initialize keys to search for
    qid = ["question_id"]
    question = ["query", "question"]
    answer = ["answer"]
    options = []
    gold = []
    aspect = []
    complexity = []

    qid_key = -1
    question_key = -1
    answer_key = -1
    options_key = -1
    gold_key = -1
    aspect_key = -1
    complexity_key = -1

    input_path = args.input
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
