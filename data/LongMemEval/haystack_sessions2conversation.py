import argparse
import json
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert LongMemEval haystack sessions to conversation json files.")
    parser.add_argument('--input', '-i', type=str, default='data/LongMemEval/longmemeval_oracle.json', help='Input haystack sessions json file')
    parser.add_argument('--output', '-o', type=str, default='data/LongMemEval/conversations/', help='Output directory for conversation json files')
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_count = 0
    for item in data:
        if not isinstance(item, dict):
            continue

        question_id = item.get('question_id')
        question_type = item.get('question_type')
        question = item.get('question')
        answer = item.get('answer')
        question_date = item.get('question_date')
        haystack_dates = item.get('haystack_dates', [])
        haystack_session_ids = item.get('haystack_session_ids', [])
        haystack_sessions = item.get('haystack_sessions', [])
        answer_session_ids = item.get('answer_session_ids', [])

        # Skip records missing critical fields
        if not question_id or not question:
            continue

        conversation = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "question_date": question_date,
            "haystack_dates": haystack_dates,
            "haystack_session_ids": haystack_session_ids,
            "haystack_sessions": haystack_sessions,
            "answer_session_ids": answer_session_ids
        }

        safe_filename = question_id + ".json"
        file_path = os.path.join(output_dir, safe_filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=4)
        
        valid_count += 1
    
    print(f"Conversion complete! Generated {valid_count} files")
    print(f"Files saved in: {output_dir}")