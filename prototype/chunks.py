import os
import pickle
from tqdm import tqdm
import argparse
from tasks.tools.utils import count_tokens
import json



def find_punctuations(text, comma=False):
    """Find punctuation indices in the text, optionally including commas."""
    if comma:
        puncs = ['.', '?', '!', ',', '."', '?"', '!"', ".'", "?'", "!'"]
    else:
        puncs = ['.', '?', '!', '."', '?"', '!"', ".'", "?'", "!'"]

    puncs_idx = []
    for i, c in enumerate(text):
        if c in puncs:
            puncs_idx.append(i)
        elif c == '"' or c == "'":
            if i > 0 and text[i-1] in ['.', '?', '!']:
                puncs_idx.append(i)

    return puncs_idx


def truncate(text, chunk_size):
    """Truncate text at safe punctuation boundaries so token count fits within chunk size."""
    ori_text = text
    ori_len = len(text)

    while count_tokens(text) > chunk_size:
        puncs_idx = find_punctuations(text)
        try:
            text = text[:puncs_idx[-2] + 1]
        except:
            puncs_idx = find_punctuations(text, comma=True)
            try:
                text = text[:puncs_idx[-2] + 1]
            except:
                # assert (ori_len - len(text)) == 0, f"Text truncation failed: {(ori_len - len(text))} characters of remaining 'truncated' part were discarded."
                # Use hard cut
                hard_cut_idx = int(len(text) * chunk_size / count_tokens(text)) - 1
                while (text[hard_cut_idx] not in [' ', '\n'] or count_tokens(text[:hard_cut_idx]) > chunk_size) and hard_cut_idx > 0:
                    hard_cut_idx -= 1
                text = text[:hard_cut_idx+1]
                return text, ori_text[len(text):]

    # new_len = len(text)
    # diff = ori_len - new_len
    truncated = ori_text[len(text):]

    return text, truncated


def chunk_text(paragraphs, chunk_size):
    """Split paragraphs into token-constrained chunks."""
    chunks = []
    curr_chunk = ''

    for p in tqdm(paragraphs, total=len(paragraphs)):
        new_chunk = '\n'.join([curr_chunk, p]) if len(curr_chunk) > 0 else p

        # if a single paragraph is too long, split it into smaller chunks
        if count_tokens(p) > chunk_size:
            curr_chunk, chunk_truncated = truncate(new_chunk, chunk_size)
            chunks.append(curr_chunk)
            while count_tokens(chunk_truncated) > chunk_size:
                curr_chunk, chunk_truncated = truncate(chunk_truncated, chunk_size)
                chunks.append(curr_chunk)
            curr_chunk = chunk_truncated
            continue

        if count_tokens(new_chunk) > chunk_size:
            chunks.append(curr_chunk)
            curr_chunk = p
        else:
            curr_chunk = new_chunk

    if len(curr_chunk) > 0:
        chunks.append(curr_chunk)

    return chunks


def process_book(title, book, chunk_size, include_empty):
    """Process a long doc into token-constrained chunks."""
    new_data = {}
    paragraphs = book.split("\n")
    index_id = 0
    if not include_empty:
        paragraphs = [p for p in paragraphs if len(p) > 0]

    print(f"processing book: {title}")
    total_size = count_tokens('\n'.join(paragraphs))
    print(f"{title} total sizes: {total_size}")

    if total_size <= chunk_size:
        new_data[str(index_id)] = {}
        new_data[str(index_id)]["title"] = f"{title}_chunk_0"
        new_data[str(index_id)]["text"] = '\n'.join(paragraphs)
    else:
        chunks = chunk_text(paragraphs, chunk_size)
        len_diff = count_tokens(''.join(paragraphs).replace('\n', '')) - count_tokens(''.join(chunks).replace('\n', ''))
        assert len_diff == 0, f"Information lost: {len_diff}"

        for chunk_id, chunk in enumerate(chunks):
            new_data[str(index_id)] = {}
            new_data[str(index_id)]["title"] = f"{title}_chunk_{chunk_id}"
            new_data[str(index_id)]["text"] = chunk
            index_id += 1

        chunk_sizes = [count_tokens(new_data[c]["text"]) for c in new_data.keys()]
        print(f"{title} chunk num: {len(chunk_sizes)}")
        print(f"{title} chunk sizes: {chunk_sizes}")
    return new_data


def chunk_conversation(paragraphs, paragraph_speakers, chunk_size):
    """Split turn texts into token-constrained chunks."""
    chunks = []
    curr_chunk = ''

    for p_idx, p in tqdm(enumerate(paragraphs), total=len(paragraphs)):
        new_turn_text = f"{paragraph_speakers[p_idx]}: {p}"
        new_chunk = '\n'.join([curr_chunk, new_turn_text]) if len(curr_chunk) > 0 else new_turn_text

        # if a single paragraph is too long, split it into smaller chunks
        if count_tokens(p) > chunk_size:
            curr_chunk, chunk_truncated = truncate(new_chunk, chunk_size)
            chunks.append(curr_chunk)
            chunk_truncated = f"{paragraph_speakers[p_idx]}: {chunk_truncated}"
            while count_tokens(chunk_truncated) > chunk_size:
                curr_chunk, chunk_truncated = truncate(chunk_truncated, chunk_size)
                chunks.append(curr_chunk)
                chunk_truncated = f"{paragraph_speakers[p_idx]}: {chunk_truncated}"
            curr_chunk = chunk_truncated
            continue

        if count_tokens(new_chunk) > chunk_size:
            chunks.append(curr_chunk)
            curr_chunk = new_turn_text
        else:
            curr_chunk = new_chunk

    if len(curr_chunk) > 0:
        chunks.append(curr_chunk)

    return chunks


def process_conversation(title, conversation, chunk_size, include_empty, mode='token'):
    """Process a conversation into token-constrained chunks."""
    new_data = {}
    chunk_id = 0

    print(f"processing conversation: {title}")

    for idx, session in enumerate(conversation.get("haystack_sessions", [])):
        date = conversation.get("haystack_dates", [])[idx] if idx < len(conversation.get("haystack_dates", [])) else "Unknown Date"
        session_text = f"Session {idx + 1} ({date}):"

        paragraphs = []
        paragraph_speakers = []
        for turn in session:
            role = turn.get('role', 'Unknown')
            content = turn.get('content', '')
            turn_text = content

            if include_empty:
                paragraphs.append(turn_text)
                paragraph_speakers.append(role)
            else:
                if content.strip():
                    paragraphs.append(turn_text)
                    paragraph_speakers.append(role)

        if not paragraphs:
            continue

        chunks = chunk_conversation(paragraphs, paragraph_speakers, chunk_size)
        len_diff = count_tokens(''.join(paragraphs).replace('\n', '')) - count_tokens(''.join(chunks).replace('\n', ''))
        assert len_diff <= 0, f"Information lost: {len_diff}"

        for chunk in chunks:
            new_data[str(chunk_id)] = {}
            new_data[str(chunk_id)]["title"] = f"{title}_chunk_{chunk_id}"
            new_data[str(chunk_id)]["text"] = f"{session_text}\n{chunk}"
            chunk_id += 1

    chunk_sizes = [count_tokens(new_data[c]["text"]) for c in new_data.keys()]
    print(f"{title} chunk num: {len(chunk_sizes)}")
    print(f"{title} chunk sizes: {chunk_sizes}")

    return new_data


def process_conversations(args):
    assert args.conversation_mode in ['token', 'turn'], "Unsupported conversation mode."

    input_dir = f'./data/{args.dataset}/conversations/'
    output_dir = f'./data/{args.dataset}/chunks/'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)

        title = os.path.splitext(filename)[0]
        chunked_data = process_conversation(title, conversation, args.chunk_size, args.include_empty_lines, mode=args.conversation_mode)

        output_path = os.path.join(output_dir, f"{title}_chunked_{args.chunk_size}.json")
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(chunked_data, out_file, indent=4)
            print(f"Saved chunked output to: {output_path}")


def main(args):
    if args.conversation_mode != 'null':
        process_conversations(args)
        return

    input_dir = f'./data/{args.dataset}/books/'
    output_dir = f'./data/{args.dataset}/chunks/'
    os.makedirs(output_dir, exist_ok=True)

    merged_chunks = {}
    global_chunk_id = 0

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue

        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            book = f.read()

        title = os.path.splitext(filename)[0]
        chunked_data = process_book(title, book, args.chunk_size, args.include_empty_lines)

        if args.multi_doc:
            for _, chunk_info in chunked_data.items():
                merged_chunks[str(global_chunk_id)] = {
                    "title": chunk_info["title"],
                    "text": chunk_info["text"],
                    "doc_id": title  # tag document origin
                }
                global_chunk_id += 1
        else:
            output_path = os.path.join(output_dir, f"{title}_chunked_{args.chunk_size}.json")
            with open(output_path, 'w', encoding='utf-8') as out_file:
                json.dump(chunked_data, out_file, indent=4)
            print(f"Saved chunked output to: {output_path}")

    if args.multi_doc:
        output_path = os.path.join(output_dir, f"merged_chunked_{args.chunk_size}.json")
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(merged_chunks, out_file, indent=4)
        print(f"Saved merged multi-document chunks to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split long documents into token-bounded chunks.")
    parser.add_argument("--dataset", type=str, default='NovelQA',
                        help="Dataset name (used to locate ./data/<dataset>/books/)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Maximum number of tokens per chunk")
    parser.add_argument("--include_empty_lines", action="store_true",
                        help="Include empty lines when splitting paragraphs")
    parser.add_argument("--multi_doc", action="store_true",
                        help="Merge all documents into a single file with doc_id tagging")
    parser.add_argument("--conversation_mode", type=str, default='null',
                        help="Process documents as conversations")
    args = parser.parse_args()
    main(args)
