import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from data.MultiHopRAG.corpus2book import sanitize_filename

def convert_longmemeval_to_books(json_file, output_dir):
    """
    将LongMemEval JSON格式数据转换为CAM的books文件格式
    每个问题及其相关信息对应一个文本文件
    """

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理每条记录
    valid_count = 0
    for item in tqdm(data, total=len(data), desc="Processing LongMemEval data..."):
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

        # 跳过缺少关键字段的记录
        if not question_id or not question:
            continue

        # 创建文件内容
        content_lines = []
        for idx_sess, session in enumerate(haystack_sessions):
            content_line = f"Session {idx_sess + 1} ({haystack_dates[idx_sess]}):\n"
            for turn in session:
                role = turn.get('role', 'Unknown')
                content = turn.get('content', '')
                content_line += f"{role}: {content}\n"

            content_lines.append(content_line)

        content = "\n".join(content_lines)

        # 创建安全的文件名
        safe_filename = question_id
        # safe_filename = sanitize_filename(safe_filename)
        safe_filename += ".txt"

        # 写入文件
        file_path = os.path.join(output_dir, safe_filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        valid_count += 1

    print(f"转换完成！共生成 {valid_count} 个文件")
    print(f"文件保存在: {output_dir}")
    return valid_count

if __name__ == "__main__":
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(workspace_dir, "LongMemEval", "longmemeval_m_cleaned.json")
    output_dir = os.path.join(workspace_dir, "LongMemEval", "books")

    valid_count = convert_longmemeval_to_books(json_file, output_dir)
