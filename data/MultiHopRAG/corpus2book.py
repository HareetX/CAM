import os
import json
import argparse
import re

# corpus2book.py
# 将 corpus.json 中的每条记录（默认读取 body 字段）保存为独立的 .txt 文件，放入 books/ 文件夹
# 用法: python corpus2book.py --input ./corpus.json --outdir ./books --field body


def sanitize_filename(name):
    # 去除非法文件名字符，限制长度
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = name.strip()
    if len(name) > 200:
        name = name[:200]
    return name or "doc"

def record_title(rec, index, default_field='body'):
    # 优先使用显式 title 字段，其次尝试使用 first line of body,最后使用索引
    if isinstance(rec, dict):
        if 'title' in rec and rec['title']:
            return rec['title']
        for k in ('headline', 'name'):
            if k in rec and rec[k]:
                return rec[k]
        if default_field in rec and rec[default_field]:
            # use first non-empty line as title
            first_line = str(rec[default_field]).splitlines()[0].strip()
            if first_line:
                return first_line[:120]
    return f"doc_{index}"

def extract_text(rec, field):
    if isinstance(rec, dict):
        if field in rec and rec[field]:
            return str(rec[field])
        # fallbacks
        for k in ('text', 'body', 'content'):
            if k in rec and rec[k]:
                return str(rec[k])
    # if record not dict or no expected fields, stringify
    return str(rec)

def main():
    parser = argparse.ArgumentParser(description="Convert corpus.json to per-document .txt files for chunks.py")
    parser.add_argument("--input", "-i", type=str, default="corpus.json", help="Path to corpus.json")
    parser.add_argument("--outdir", "-o", type=str, default="books", help="Output directory for .txt files")
    parser.add_argument("--field", "-f", type=str, default="body", help="JSON field to use as document text (default: body)")
    args = parser.parse_args()

    input_path = args.input
    out_dir = args.outdir
    field = args.field

    os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list at top-level in {input_path}")

    used_names = {}
    for idx, rec in enumerate(data):
        text = extract_text(rec, field)
        if not text or text.strip() == "":
            # skip empty documents
            continue

        raw_title = record_title(rec, idx, default_field=field)
        name = sanitize_filename(raw_title)
        # avoid duplicate names
        if name in used_names:
            used_names[name] += 1
            name = f"{name}_{used_names[name]}"
        else:
            used_names[name] = 0

        out_path = os.path.join(out_dir, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as outf:
            outf.write(text)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
