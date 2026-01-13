import os
from tasks.tools.prompts import (
    gist_generation_template,
    statement_extraction_template,
    entity_extraction_template,
    kg_extraction_template
)
from tqdm import tqdm
import numpy as np
import json
import argparse
from tasks.tools.utils import APIClient
from concurrent.futures import ThreadPoolExecutor, as_completed



def _extract_statements_from_chunk(chunk, client):
    es_prompt = statement_extraction_template.format(input_chunk=chunk)
    es_response = client.obtain_response(es_prompt, max_tokens=1024, temperature=0.0, json_output=True).strip()
    es_response = json.loads(es_response).get("statements", [])
    es_response = [stmt.strip() for stmt in es_response if stmt.strip()]
    return es_response


def _extract_kg_from_chunk(chunk, client):
    ekg_prompt = kg_extraction_template.format(input_chunk=chunk)
    ekg_response = client.obtain_response(ekg_prompt, max_tokens=16384, temperature=0.0, json_output=True).strip()
    try:
        ekg_response = json.loads(ekg_response).get("graph_elements", {"entities": [], "relationships": []})
    except json.JSONDecodeError:
        # print(ekg_response)
        ekg_response = {"entities": [], "relationships": []}
    return ekg_response


def process_single_doc(filename, file_path, client, dataset, generate_gist, extract_entity, conversation_mode):
    stmt_modes = ['statement']
    kg_modes = ['kg', 'kg_node_cluster_wo_kg', 'kg_edge_cluster_wo_kg']

    file_title = os.path.splitext(filename)[0]

    output_embedding_path = f'./processed_data/{dataset}/chunk_embeddings/'
    output_metadata_path = f'./processed_data/{dataset}/chunk_metadata/'
    output_embedding = os.path.join(output_embedding_path, f'{file_title}_embeddings.npy')
    output_metadata = os.path.join(output_metadata_path, f'{file_title}_metadata.json')
    output_statement_path = f'./processed_data/{dataset}/chunk_statements/'
    output_kg_path = f'./processed_data/{dataset}/chunk_kg/'

    if os.path.exists(output_embedding) and os.path.exists(output_metadata):
        print(f"Skipping {filename}...")
        return

    print(f"Processing {filename}...")

    with open(os.path.join(file_path, filename), 'r') as f:
        data = json.load(f)

    all_chunk_dict_list = []
    all_chunk_statement_list = []
    all_chunk_kg_list = []
    all_text_embedding_list = []

    for idx, key in enumerate(tqdm(data, desc=f"Chunks in {filename}")):
        chunk = data[key]['text']
        gg_response = ""
        es_response = []
        stmt_embeddings = []
        ee_response = []

        if generate_gist:
            gg_prompt = gist_generation_template.format(input_chunk=chunk)
            gg_response = client.obtain_response(gg_prompt, max_tokens=500, temperature=0.0).strip()

        # extract statements from each chunk if conversation_mode is set
        if conversation_mode in stmt_modes:
            es_response = _extract_statements_from_chunk(chunk, client)
            stmt_embeddings = []
            for stmt in es_response:
                stmt_embedding = client.obtain_embedding(stmt)
                stmt_embedding = np.array(stmt_embedding, dtype=np.float32)
                stmt_embeddings.append(stmt_embedding)

        # extract kg entities and relations from each chunk if conversation_mode is set
        if conversation_mode in kg_modes:
            ekg_response = _extract_kg_from_chunk(chunk, client)

        # extract entity
        if extract_entity:
            ee_prompt = entity_extraction_template.format(input_chunk=chunk)
            ee_response = client.obtain_response(ee_prompt, max_tokens=500, temperature=0.0).strip()
            ee_response = [entity.strip() for entity in ee_response.split(';')]
            ee_response = list(dict.fromkeys(ee_response))[:5]
            ee_response = [s for s in ee_response if s]

        # embedding
        chunk_embedding = client.obtain_embedding(chunk)
        chunk_embedding = np.array(chunk_embedding, dtype=np.float32)

        # doc ids
        doc_id = data[key].get("doc_id")
        if not doc_id:
            # single-doc: <title>_chunked_<size>.json
            doc_id = filename.split("_chunked")[0]

        # chunk dict
        chunk_dict = {
            "chunk_id": idx,
            "text": chunk,
            "gist": gg_response,
            "entity_concepts": ee_response,
            "doc_id": doc_id
        }
        all_chunk_dict_list.append(chunk_dict)

        # chunk statement dict
        if conversation_mode in stmt_modes:
            chunk_statement_dict = {
                "chunk_id": idx,
                "statements": es_response,
                "statement_embeddings": stmt_embeddings
            }
            all_chunk_statement_list.append(chunk_statement_dict)

        # chunk kg dict
        if conversation_mode in kg_modes:
            chunk_kg_dict = {
                "chunk_id": idx,
                "graph_elements": ekg_response
            }
            all_chunk_kg_list.append(chunk_kg_dict)

        # chunk embedding
        all_text_embedding_list.append(chunk_embedding)

    embeddings = np.stack(all_text_embedding_list, axis=0)

    os.makedirs(output_embedding_path, exist_ok=True)
    np.save(output_embedding, embeddings)

    os.makedirs(output_metadata_path, exist_ok=True)
    with open(output_metadata, "w") as f:
        json.dump(all_chunk_dict_list, f, indent=4)

    if conversation_mode in stmt_modes:
        output_statement_metadata_path = os.path.join(output_statement_path, 'metadata')
        output_statement_embedding_path = os.path.join(output_statement_path, 'embeddings')
        output_statement_metadata = os.path.join(output_statement_metadata_path, f'{file_title}_statements.json')
        os.makedirs(output_statement_path, exist_ok=True)
        os.makedirs(output_statement_metadata_path, exist_ok=True)
        os.makedirs(output_statement_embedding_path, exist_ok=True)
        # statement embeddings
        for idx, chunk_stmt in enumerate(all_chunk_statement_list):
            chunk_id = chunk_stmt['chunk_id']
            stmt_embs = chunk_stmt['statement_embeddings']
            output_statement_embedding = os.path.join(output_statement_embedding_path, f'{file_title}_chunk_{chunk_id}_stmt_embeddings.npy')
            np.save(output_statement_embedding, np.stack(stmt_embs, axis=0))
            # delete statement embeddings to save spac
            all_chunk_statement_list[idx].pop('statement_embeddings', None)
        # statement metadata
        with open(output_statement_metadata, "w") as f:
            json.dump(all_chunk_statement_list, f, indent=4)

    if conversation_mode in kg_modes:
        output_kg_metadata_path = os.path.join(output_kg_path, 'metadata')
        os.makedirs(output_kg_path, exist_ok=True)
        os.makedirs(output_kg_metadata_path, exist_ok=True)
        output_kg_metadata = os.path.join(output_kg_metadata_path, f'{file_title}_kg.json')
        # kg metadata
        with open(output_kg_metadata, "w") as f:
            json.dump(all_chunk_kg_list, f, indent=4)

    print(f"Finished {filename}: {len(all_chunk_dict_list)} chunks, embedding shape {np.stack(embeddings).shape}")


def process_multi_doc(file_path, filename, client, dataset, generate_gist, extract_entity):
    print(f"Processing merged file: {filename} [multi-doc mode]")
    with open(os.path.join(file_path, filename), 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_chunk_dicts = []
    all_embeddings = []

    for idx, key in enumerate(tqdm(data, desc=f"Chunks in {filename}")):
        chunk = data[key]['text']
        gg_response = ""
        ee_response = []

        if generate_gist:
            gg_prompt = gist_generation_template.format(input_chunk=chunk)
            gg_response = client.obtain_response(gg_prompt, max_tokens=500, temperature=0.0).strip()

        # extract entity
        if extract_entity:
            ee_prompt = entity_extraction_template.format(input_chunk=chunk)
            ee_response = client.obtain_response(ee_prompt, max_tokens=500, temperature=0.0).strip()
            ee_response = [entity.strip() for entity in ee_response.split(';')]
            ee_response = list(dict.fromkeys(ee_response))[:5]
            ee_response = [s for s in ee_response if s]

        # embedding
        chunk_embedding = client.obtain_embedding(chunk)
        chunk_embedding = np.array(chunk_embedding, dtype=np.float32)

        chunk_dict = {
            "chunk_id": int(idx),
            "text": chunk,
            "gist": gg_response,
            "entity_concepts": ee_response,
            "doc_id": data[key].get("doc_id")
        }

        all_chunk_dicts.append(chunk_dict)
        all_embeddings.append(chunk_embedding)

    os.makedirs(f'./processed_data/{dataset}/chunk_embeddings', exist_ok=True)
    os.makedirs(f'./processed_data/{dataset}/chunk_metadata', exist_ok=True)

    np.save(f'./processed_data/{dataset}/chunk_embeddings/merged_embeddings.npy', np.stack(all_embeddings))
    with open(f'./processed_data/{dataset}/chunk_metadata/merged_metadata.json', 'w') as f:
        json.dump(all_chunk_dicts, f, indent=4)

    doc_ids = [chunk['doc_id'] for chunk in all_chunk_dicts]
    N = len(doc_ids)
    doc_mask = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        for j in range(N):
            if doc_ids[i] == doc_ids[j]:
                doc_mask[i, j] = 1
    np.save(f'./processed_data/{dataset}/chunk_embeddings/merged_doc_mask.npy', doc_mask)

    print(f"Saved merged doc mask: shape {doc_mask.shape}")


def main():
    parser = argparse.ArgumentParser(description="Chunk-level gist/entity/embedding extraction")
    parser.add_argument("--dataset", type=str, default='NovelQA',
                        help="Dataset name")
    parser.add_argument("--multi_doc", action="store_true",
                        help="Enable multi-document processing")
    parser.add_argument("--api_key_path", type=str, default="openai_key.txt",
                        help="Path to OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-large",
                        help="Embedding model to use")
    parser.add_argument("--generate_gist", action="store_true",
                        help="If set, generate a gist for each chunk")
    parser.add_argument("--extract_entity", action="store_true",
                        help="If set, extract entities from each chunk")
    parser.add_argument("--max_workers", type=int, default=3,
                        help="Max parallel workers for single-doc mode")
    parser.add_argument("--merged_filename", type=str, default="merged_chunked_512.json",
                        help="Merged file name produced by chunk.py in multi-doc mode")
    parser.add_argument("--conversation_mode", type=str, default="null",
                        help="Process documents as conversations")
    args = parser.parse_args()

    client = APIClient("openai", args.api_key_path, args.model, args.embedding_model)
    file_path = f'./data/{args.dataset}/chunks/'

    if args.multi_doc:
        process_multi_doc(
            file_path=file_path,
            filename=args.merged_filename,
            client=client,
            dataset=args.dataset,
            generate_gist=args.generate_gist,
            extract_entity=args.extract_entity
        )
    else:
        files = [f for f in os.listdir(file_path) if f.endswith('.json')]
        max_workers = max(1, min(args.max_workers, len(files)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_doc, filename, file_path, client, args.dataset, args.generate_gist, args.extract_entity, args.conversation_mode): filename for filename in files}

            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()
