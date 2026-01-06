import argparse
import os
import json

QID = "QID"
QUESTION = "Question"
ANSWER = "Answer"
OPTIONS = "Options"
GOLD = "Gold"
ASPECT = "Aspect"
COMPLEXITY = "Complexity"

def main():
    parser = argparse.ArgumentParser(description="Convert corpus.json to per-document .txt files for chunks.py")
    parser.add_argument("--input", "-i", type=str, default="data/LongMemEval/longmemeval_oracle.json", help="Path to json file")
    parser.add_argument("--outdir", "-o", type=str, default="data/LongMemEval/questions", help="Output directory for json files")
    args = parser.parse_args()

    # Initialize keys to search for
    qid_key = "question_id"
    question_key = "question"
    answer_key = "answer"
    options_key = -1
    gold_key = -1
    aspect_key = -1
    complexity_key = -1

    input_path = args.input
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r") as f:
        data = json.load(f)
        for item in data:
            qid_value = item[qid_key]
            question_value = item[question_key]
            answer_value = item[answer_key]
            options_value = item[options_key] if options_key != -1 else None
            gold_value = item[gold_key] if gold_key != -1 else None
            aspect_value = item[aspect_key] if aspect_key != -1 else None
            complexity_value = item[complexity_key] if complexity_key != -1 else None

            out_item = {
                QID: str(qid_value),
                QUESTION: str(question_value),
                ANSWER: str(answer_value)
            }
            if options_value is not None:
                out_item[OPTIONS] = str(options_value)
            if gold_value is not None:
                out_item[GOLD] = str(gold_value)
            if aspect_value is not None:
                out_item[ASPECT] = str(aspect_value)
            if complexity_value is not None:
                out_item[COMPLEXITY] = str(complexity_value)

            out_path = os.path.join(out_dir, f"{qid_value}.json")
            with open(out_path, "w") as out_f:
                json.dump([out_item], out_f, indent=4)

    print(f"Converted {len(data)} entries from {input_path} to individual json files in {out_dir}")


if __name__ == "__main__":
    main()
