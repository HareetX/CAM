import os
import random
from tqdm import tqdm
import numpy as np
import warnings
import json
import argparse
from tasks.tools.utils import APIClient
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tasks.tools.prompts import text_summarization_template
from itertools import combinations
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    p = argparse.ArgumentParser(description="One-shot Constructivist Agentic Memory with hierarchy (parallel single-doc).")
    p.add_argument("--dataset", type=str, default="NovelQA", help="Dataset name.")
    p.add_argument("--chunk_size", type=int, default=512, help="Chunk size used in chunking (for filename resolution).")
    p.add_argument("--threshold", type=float, default=0.6, help="Edge activation threshold.")
    p.add_argument("--weight", type=float, default=0.6, help="Weight for text similarity vs proximity (0~1).")
    p.add_argument("--sigma", type=float, default=1.0, help="Sigma for Gaussian proximity similarity.")
    p.add_argument("--k", type=int, default=10, help="Top-k neighbors per node.")
    p.add_argument("--multi_doc", action="store_true", help="Use merged multi-document inputs and doc mask.")
    p.add_argument("--max_cluster_size", type=int, default=12, help="Maximum nodes allowed in one community.")
    p.add_argument("--api_key_path", type=str, default="openai_key.txt", help="Path to OpenAI API key.")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model for summarization.")
    p.add_argument("--embedding_model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    p.add_argument("--max_hierarchy_level", type=int, default=10, help="Maximum hierarchy levels.")
    p.add_argument("--summary_field", type=str, default="text", help="Node attribute to use as text for hierarchical summarization.")
    p.add_argument("--num_processes", type=int, default=1, help="Processes for single-doc parallel build (default=min(10,CPU)).")
    p.add_argument("--sample_num", type=int, default=10, help="Number of files to sample for single-doc testing.")
    p.add_argument("--num_workers_cpu", type=int, default=None, help="Workers for CPU-bound tasks in hierarchy building (None=single-processed).")
    p.add_argument("--num_workers_io", type=int, default=None, help="Workers for I/O-bound tasks in hierarchy building (None=single-threaded).")
    p.add_argument("--conversation_mode", type=str, default="null", help="Conversation mode for single-doc chunk preprocessing (e.g., 'null', 'token', 'turn', 'statement').")
    return p.parse_args()

class CAM:
    def __init__(self,
                dataset: str,
                threshold: float = 0.6,
                weight: float = 0.6,
                sigma: float = 1.0,
                top_k: int = 10,
                api_key_path: str = "openai_key.txt",
                model: str = "gpt-4o-mini",
                embedding_model: str = "text-embedding-3-large",
                max_cluster_size: int = 12,
                summary_field: str = "text"):

        self.dataset = dataset
        self.activation_threshold = threshold
        self.w_text = weight
        self.sigma = sigma
        self.top_k = top_k
        self.max_cluster_size = max_cluster_size
        self.summary_field = summary_field

        self.client = APIClient("openai", api_key_path, model, embedding_model)
        self.client_args = {
            "api_key_path": api_key_path,
            "model": model,
            "embedding_model": embedding_model
        }

        self.memory = nx.Graph()
        self.embeddings = None                  # [N, D]
        self.doc_mask = None                    # [N, N], only used in multi-doc
        self.node_ids = []
        self.doc_ids = []

    def _prepare_output_dirs(self, book_title):
        self.book_title = book_title
        self.graph_out_dir = f"./super_graphs/{self.dataset}/{book_title}"
        self.emb_out_dir = f"./super_embeddings/{self.dataset}/{book_title}"
        os.makedirs(self.graph_out_dir, exist_ok=True)
        os.makedirs(self.emb_out_dir, exist_ok=True)

    def _save_level_graph_and_embeddings(self, graph: nx.Graph, embeddings: np.ndarray, level: int):
        """Save graph & embeddings for a specific level; also remember them in level arrays."""
        # Keep in memory for composing later
        self.level_graphs.append(graph.copy())
        self.level_embeddings.append(embeddings.copy() if embeddings is not None else None)

        # Persist to disk
        if embeddings is not None:
            np.save(os.path.join(self.emb_out_dir, f"{self.book_title}_embedding_level_{level}.npy"), embeddings)

        G = graph.copy()
        for node, data in G.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list):
                    G.nodes[node][k] = str(v)
        nx.write_gexf(G, os.path.join(self.graph_out_dir, f"{self.book_title}_graph_level_{level}.gexf"))

    def _save_graph_with_community(self, graph, community_dict, level):
        """Save graph with 'community' node attribute (for Gephi inspection)."""
        G = graph.copy()
        nx.set_node_attributes(G, community_dict, "community")
        for node, data in G.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list):
                    G.nodes[node][k] = str(v)
        out_path = os.path.join(self.graph_out_dir, f"{self.book_title}_graph_with_community_{level}.gexf")
        nx.write_gexf(G, out_path)

    def _add_nodes_from_metadata(self, metadata: list[dict], title_for_default_doc: str):
        """
        Metadata list entries should contain:
        - chunk_id (int)
        - text (str)
        - gist (str)
        - entity_concepts (list[str])
        - doc_id (str): optional, used in multi-doc
        """
        N = len(metadata)
        self.node_ids = list(range(N))
        self.doc_ids = []
        for i, item in enumerate(metadata):
            doc_id = item.get("doc_id", title_for_default_doc)
            self.doc_ids.append(doc_id)
            self.memory.add_node(
                i,
                chunk_id=item.get("chunk_id"),
                text=item.get("text", ""),
                gist=item.get("gist", ""),
                entities=[e for e in item.get("entity_concepts", []) if e],
                doc_id=doc_id
            )

    def _add_nodes_from_stmt_metadata(self, metadata: list[dict], stmt_metadata: list[dict], title_for_default_doc: str):
        """
        Metadata list entries should contain:
        - chunk_id (int)
        - text (str)
        - gist (str)
        - entity_concepts (list[str])
        - doc_id (str): optional, used in multi-doc
        """
        N = 0
        for item in stmt_metadata:
            N += len(item.get("statements", []))
        self.node_ids = list(range(N))
        self.doc_ids = []
        i = 0
        for idx, item in enumerate(stmt_metadata):
            doc_id = item.get("doc_id", title_for_default_doc)
            self.doc_ids.append(doc_id)
            entity_concepts = metadata[idx].get("entity_concepts", [])
            for stmt in item.get("statements", []):
                self.memory.add_node(
                    i,
                    chunk_id=item.get("chunk_id"),
                    text=stmt,
                    gist=stmt,
                    entities=[e for e in entity_concepts if e], # reuse chunk-level entities for now
                    doc_id=doc_id
                )
                i += 1

    def _build_pairwise_similarity(self):
        """
        Build N x N similarity with text + proximity.
        In multi-doc mode, proximity is masked by doc_mask (cross-doc = 0).
        """
        N = len(self.node_ids)
        if self.embeddings is None or self.embeddings.shape[0] != N:
            raise ValueError("Embeddings not loaded or mismatched with nodes.")

        # Textual similarity
        text_sim = cosine_similarity(self.embeddings, self.embeddings)
        np.fill_diagonal(text_sim, 0.0)
        text_sim = np.maximum(0, text_sim)

        # Proximity based on chunk_id (within the same doc)
        chunk_ids = np.array([self.memory.nodes[n]['chunk_id'] for n in self.node_ids], dtype=np.int64)
        diff = np.abs(chunk_ids[:, None] - chunk_ids[None, :])
        proximity_sim = np.exp(- (diff ** 2) / (2 * (self.sigma ** 2)))
        np.fill_diagonal(proximity_sim, 0.0)

        # If doc_mask is present (multi-doc), zero out cross-doc proximity
        if self.doc_mask is not None:
            if self.doc_mask.shape != (N, N):
                raise ValueError(f"doc_mask shape {self.doc_mask.shape} mismatches N={N}.")
            proximity_sim *= self.doc_mask

        # Weighted combination
        sim = self.w_text * text_sim + (1.0 - self.w_text) * proximity_sim
        return sim

    def _add_edges_from_similarity(self, sim: np.ndarray):
        """
        For each node, connect to top-k neighbors above threshold.
        """
        N = sim.shape[0]
        for i in range(N):
            sims = sim[i]
            above = np.where(sims >= self.activation_threshold)[0]
            if above.size == 0:
                continue
            # stable sort by descending similarity
            top_idx = above[np.argsort(-sims[above], kind='stable')][:min(self.top_k, above.size)]
            for j in top_idx:
                if i != j:
                    self.memory.add_edge(i, j, weight=float(sims[j]))
        self.memory.remove_edges_from(nx.selfloop_edges(self.memory))

    def _limit_community_size(self, communities):
        """Split large communities if exceeding max_cluster_size."""
        if not self.max_cluster_size:
            return communities
        new_comms = []
        for comm in communities:
            comm = list(comm)
            if len(comm) <= self.max_cluster_size:
                new_comms.append(set(comm))
            else:
                for i in range(0, len(comm), self.max_cluster_size):
                    new_comms.append(set(comm[i:i + self.max_cluster_size]))
        return new_comms

    def _create_persona_graph(self, graph: nx.Graph):
        components = {}
        personalities = {}
        index = 0

        # Step 1: Create egonets and partition them
        for node in graph.nodes():
            # Egonet minus ego: subgraph of neighbors
            if not list(graph.neighbors(node)):
                personas = [index]
                index += 1
                personalities[node] = personas
                continue
            ego_net_minus_ego = graph.subgraph(graph.neighbors(node))
            comps = list(nx.connected_components(ego_net_minus_ego))
            new_mapping = {}
            personas = []
            for comp in comps:
                personas.append(index)
                for other_node in comp:
                    new_mapping[other_node] = index
                index += 1
            components[node] = new_mapping  # {node: {neighbor: persona_id}}
            personalities[node] = personas  # {node: [persona_ids]}

        # Step 2: Map edges to persona graph
        persona_graph_edges = []
        for u, v in graph.edges():
            if v in components[u] and u in components[v]:
                persona_u_v = components[u][v]  # Persona of v in u's egonet
                persona_v_u = components[v][u]  # Persona of u in v's egonet
                weight = graph[u][v].get('weight', 1.0)
                persona_graph_edges.append((persona_u_v, persona_v_u, {'weight': weight}))

        # Step 3: Create the persona graph
        persona_graph = nx.Graph()
        persona_graph.add_nodes_from(range(index))
        persona_graph.add_edges_from(persona_graph_edges)

        personality_map = {p: n for n in graph.nodes() for p in personalities[n]} # {persona_id: original_node_id}

        return persona_graph, components, personalities, personality_map

    def _build_super_graph(self, graph, communities, community_dict):
        super_graph = nx.Graph()
        super_graph.add_nodes_from(range(len(communities)))
        # overlap edges
        for idx1, idx2 in combinations(range(len(communities)), 2):
            if communities[idx1] & communities[idx2]:
                super_graph.add_edge(idx1, idx2)
        # non-overlap edges
        for (u, v) in graph.edges():
            for comm1 in community_dict[u]:
                for comm2 in community_dict[v]:
                    if comm1 != comm2:
                        super_graph.add_edge(comm1, comm2)
        return super_graph

    def _text_summarization(self, texts: list[str]):
        """LLM-based community summarization."""
        input_texts = "\n".join(f"Passage {i+1}: {text}" for i, text in enumerate(texts))
        prompt = text_summarization_template.format(input_texts=input_texts)
        response = self.client.obtain_response(prompt, max_tokens=1024, temperature=0.0)
        super_gist = response.strip()
        super_emb = np.array(self.client.obtain_embedding(super_gist), dtype=np.float32)
        return super_gist, super_emb

    def _print_community_stats(self, level: int, book_title: str, graph: nx.Graph, communities: list[set]):
        sizes = [len(c) for c in communities]
        num = len(sizes)
        if num == 0:
            print(f"[Level {level}] No communities found.")
            return
        max_size = max(sizes)
        min_size = min(sizes)
        avg_size = sum(sizes) / num

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        print(f"[Level {level} of {book_title}] Nodes: {num_nodes} | Edges: {num_edges} | Avg Degree: {avg_degree:.2f}")
        print(f"[Level {level} of {book_title}] Communities: {num} | Max size: {max_size} | Min size: {min_size} | Avg size: {avg_size:.2f}")

    def _parallel_label_propagation(self, graph: nx.Graph, num_workers: int):
        """Parallel version of label propagation community detection."""

        # Inner function to process a subgraph
        def process_subgraph(subgraph):
            return list(nx.community.label_propagation_communities(subgraph))

        # Split graph into connected components for parallel processing
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        # Use multiprocessing Pool to process components in parallel
        final_communities = []
        with Pool(processes=num_workers) as pool:
            subgraph_communities = pool.map(process_subgraph, components)

        # Combine results
        for comms in subgraph_communities:
            final_communities.extend(comms)

        return final_communities

    def _parallel_summarize_communities(self, communities, current_graph, num_threads: int, level: int, book_title: str):
        """Parallel summarization of communities. (I/O bound)"""
        # Prepare inputs
        texts_list = []
        for comm in communities:
            node_list_sorted = sorted(list(comm))
            texts = [current_graph.nodes[n].get(self.summary_field) for n in node_list_sorted]
            texts_list.append(texts)

        # Use ThreadPool for I/O bound tasks
        new_texts = []
        new_embs = []

        with ThreadPool(processes=num_threads) as pool:
            results = list(tqdm(
                pool.imap(self._text_summarization, texts_list),
                total=len(texts_list),
                desc=f"Summarizing level {level} communities for {book_title}"
            ))

        for super_gist, emb in results:
            new_texts.append(super_gist)
            new_embs.append(emb)
        return new_texts, new_embs

    def run_hierarchy(self, book_title, max_hierarchy_level: int = 10, num_workers_cpu: int = None, num_workers_io: int = None):

        self.level_graphs = []
        self.level_embeddings = []

        # Level 0: current memory state
        self._save_level_graph_and_embeddings(graph=self.memory, embeddings=self.embeddings, level=0)

        current_graph = self.memory.copy()
        prev_num_communities = current_graph.number_of_nodes()
        level = 0

        while True:
            persona_graph, _, _, personality_map = self._create_persona_graph(current_graph)
            communities = []
            if num_workers_cpu is None:
                communities = list(nx.community.label_propagation_communities(persona_graph))
            else:
                communities = self._parallel_label_propagation(persona_graph, num_workers=num_workers_cpu)
            communities = self._limit_community_size(communities)

            # map persona communities back to original nodes
            overlap_communities = [] # list of sets of original node ids for each community
            community_dict = {} # node_id -> [community_ids]
            for idx, comm in enumerate(communities):
                original_nodes = {personality_map[pid] for pid in comm}
                overlap_communities.append(original_nodes)
                for n in original_nodes:
                    community_dict.setdefault(n, []).append(idx)

            self._print_community_stats(level, book_title, current_graph, overlap_communities)
            self._save_graph_with_community(current_graph, community_dict, level)

            current_num_communities = len(overlap_communities)
            if level >= max_hierarchy_level or current_num_communities >= prev_num_communities:
                break

            level += 1

            # Build super graph for next level
            super_graph = self._build_super_graph(current_graph, overlap_communities, community_dict)
            node_comm_dict = {idx: sorted(list(comm)) for idx, comm in enumerate(overlap_communities)} # community_id -> [original_node_ids]
            nx.set_node_attributes(super_graph, node_comm_dict, "community")

            # Summarize chosen field per community
            new_texts = []
            new_embs = []
            if num_workers_io is None:
                for comm in tqdm(overlap_communities, desc=f"Summarizing level {level-1} communities for {book_title}"):
                    node_list_sorted = sorted(list(comm))
                    texts = [current_graph.nodes[n].get(self.summary_field) for n in node_list_sorted]
                    super_gist, emb = self._text_summarization(texts)
                    new_texts.append(super_gist)
                    new_embs.append(emb)
            else:
                # Parallel summarization
                new_texts, new_embs = self._parallel_summarize_communities(overlap_communities, current_graph, num_threads=num_workers_io, level=level, book_title=book_title)
            new_embs = np.stack(new_embs, axis=0)
            nx.set_node_attributes(super_graph, {n: txt for n, txt in zip(super_graph.nodes(), new_texts)}, self.summary_field)

            # Save this level graph+embeddings (store level-wise, do NOT overwrite base)
            self._save_level_graph_and_embeddings(graph=super_graph, embeddings=new_embs, level=level)

            # Prepare next iteration
            prev_num_communities = current_num_communities
            current_graph = super_graph

        print(f"[Hierarchy] Completed up to level {level}.")

        # Compose ALL levels into a single graph and embeddings
        self._compose_all_levels()

    def _compose_all_levels(self):

        composed_graphs = []
        composed_embeddings = []
        offset = 0
        prev_offset = 0
        level_offsets = []  # record start offset of each level (to convert community ids)

        for level_idx, (g_level, emb_level) in enumerate(zip(self.level_graphs, self.level_embeddings)):
            start_offset = offset
            level_offsets.append(start_offset)

            # relabel nodes
            g = g_level.copy()
            mapping = {node: node + start_offset for node in g.nodes()}
            g = nx.relabel_nodes(g, mapping)

            # annotate
            nx.set_node_attributes(g, level_idx, "level")
            nx.set_node_attributes(g, start_offset, "offset")
            nx.set_node_attributes(g, prev_offset, "prev_offset")

            # convert community -> global ids (only for level >= 1)
            if level_idx >= 1:
                prev_level_offset = level_offsets[level_idx - 1]
                for node in g.nodes():
                    orig_node = node - start_offset  # reverse mapping to original id in g_level
                    comm_raw = g_level.nodes[orig_node].get("community")
                    comm_conv = [int(c) + prev_level_offset for c in comm_raw]
                    g.nodes[node]["community"] = comm_conv

            composed_graphs.append(g)
            composed_embeddings.append(emb_level)

            prev_offset = start_offset
            offset += g.number_of_nodes()

        # Compose graphs & concat embeddings
        G_all = nx.compose_all(composed_graphs)
        E_all = np.concatenate(composed_embeddings, axis=0)

        # Save
        G_safe = G_all.copy()
        for node, data in G_safe.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list):
                    G_safe.nodes[node][k] = str(v)

        nx.write_gexf(G_safe, os.path.join(self.graph_out_dir, f"{self.book_title}_graph_level_all.gexf"))
        np.save(os.path.join(self.emb_out_dir, f"{self.book_title}_embedding_level_all.npy"), E_all)

        # Expose as current memory
        self.memory = G_all
        self.embeddings = E_all
        print(f"[Compose] All levels composed: nodes={self.memory.number_of_nodes()}, edges={self.memory.number_of_edges()}")

    def build_memory(self,
                   book_title,
                   metadata,
                   embeddings,
                   doc_mask,
                   max_hierarchy_level: int = 10,
                   num_workers_cpu: int = None,
                   num_workers_io: int = None):

        self._prepare_output_dirs(book_title)

        # Add nodes
        self._add_nodes_from_metadata(metadata, title_for_default_doc=book_title)
        self.embeddings = np.atleast_2d(embeddings).astype(np.float32)
        self.doc_mask = doc_mask  # None for single-doc; (N,N) for multi-doc

        # Build edges from pairwise similarity for level-0
        sim = self._build_pairwise_similarity()
        self._add_edges_from_similarity(sim)

        # Run hierarchy & compose all levels
        self.run_hierarchy(book_title, max_hierarchy_level=max_hierarchy_level, num_workers_cpu=num_workers_cpu, num_workers_io=num_workers_io)

    def build_memory_stmt(self,
                          book_title,
                          metadata,
                          embeddings,
                          stmt_metadata,
                          stmt_embeddings,
                          doc_mask,
                          max_hierarchy_level: int = 10,
                          num_workers_cpu: int = None,
                          num_workers_io: int = None,
                          conversation_mode: str = "statement"):
        assert conversation_mode is not None, "[Error] conversation_mode must be specified for statement-level memory."

        self._prepare_output_dirs(book_title)

        # Add nodes
        if conversation_mode is not None:
            self._add_nodes_from_stmt_metadata(metadata, stmt_metadata, title_for_default_doc=book_title)
            # flatten stmt_embeddings
            all_stmt_embs = []
            for emb in stmt_embeddings:
                all_stmt_embs.extend(emb)
            self.embeddings = np.atleast_2d(np.array(all_stmt_embs)).astype(np.float32)
        else:
            self._add_nodes_from_metadata(metadata, title_for_default_doc=book_title)
            self.embeddings = np.atleast_2d(embeddings).astype(np.float32)
        self.doc_mask = doc_mask  # None for single-doc; (N,N) for multi-doc

        # Build edges from pairwise similarity for level-0
        sim = self._build_pairwise_similarity()
        self._add_edges_from_similarity(sim)

        # Run hierarchy & compose all levels
        self.run_hierarchy(book_title, max_hierarchy_level=max_hierarchy_level, num_workers_cpu=num_workers_cpu, num_workers_io=num_workers_io)

def load_json(fp: str):
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def load_single_doc(dataset: str, chunk_file_stem: str):

    meta_fp = f'./processed_data/{dataset}/chunk_metadata/{chunk_file_stem}_metadata.json'
    emb_fp = f'./processed_data/{dataset}/chunk_embeddings/{chunk_file_stem}_embeddings.npy'

    meta = load_json(meta_fp)
    emb = np.load(emb_fp)

    return meta, emb

def load_statements(dataset: str, chunk_file_stem: str):

    stmt_meta_fp = f'./processed_data/{dataset}/chunk_statements/metadata/{chunk_file_stem}_statements.json'
    stmt_emb_dir = f'./processed_data/{dataset}/chunk_statements/embeddings/'

    stmt_meta = load_json(stmt_meta_fp)
    stmt_embs = []
    for chunk_stmt in stmt_meta:
        chunk_id = chunk_stmt['chunk_id']
        stmt_emb_fp = os.path.join(stmt_emb_dir, f'{chunk_file_stem}_chunk_{chunk_id}_stmt_embeddings.npy')
        emb = np.load(stmt_emb_fp)
        stmt_embs.append(emb)
    return stmt_meta, stmt_embs

def load_merged_doc(dataset: str):
    meta_fp = f'./processed_data/{dataset}/chunk_metadata/merged_metadata.json'
    emb_fp = f'./processed_data/{dataset}/chunk_embeddings/merged_embeddings.npy'
    mask_fp = f'./processed_data/{dataset}/chunk_embeddings/merged_doc_mask.npy'

    meta = load_json(meta_fp)
    emb = np.load(emb_fp)
    mask = np.load(mask_fp)

    return meta, emb, mask

def single_doc_worker(filename,
                      dataset,
                      chunk_size,
                      cam_kwargs,
                      max_hierarchy_level,
                      num_workers_cpu,
                      num_workers_io,
                      conversation_mode='null'):
    # Only 'statement' mode needs special handling here
    safe_conversation_mode = None
    stmt_modes = ['statement']
    if conversation_mode in stmt_modes:
        safe_conversation_mode = conversation_mode

    try:
        # filename: <book>_chunked_<chunk_size>.json
        title_stem = os.path.splitext(filename)[0]  # without .json
        book_title = title_stem.split(f"_chunked_{chunk_size}")[0]

        metadata, embeddings = load_single_doc(dataset, title_stem)
        if safe_conversation_mode is not None:
            # load statement-level metadata & embeddings
            stmt_meta, stmt_embs = load_statements(dataset, title_stem)

        cam = CAM(**cam_kwargs)
        if safe_conversation_mode is None:
            cam.build_memory(book_title=book_title,
                             metadata=metadata,
                             embeddings=embeddings,
                             doc_mask=None,
                             max_hierarchy_level=max_hierarchy_level,
                             num_workers_cpu=num_workers_cpu,
                             num_workers_io=num_workers_io)
        else:
            cam.build_memory_stmt(book_title=book_title,
                                  metadata=metadata,
                                  embeddings=embeddings,
                                  stmt_metadata=stmt_meta,
                                  stmt_embeddings=stmt_embs,
                                  doc_mask=None,
                                  max_hierarchy_level=max_hierarchy_level,
                                  num_workers_cpu=num_workers_cpu,
                                  num_workers_io=num_workers_io,
                                  conversation_mode=safe_conversation_mode)
        return (book_title, True, "ok")
    except Exception as e:
        return (filename, False, str(e))

def main():
    random.seed(42)

    args = parse_args()

    # Only the multi_doc mode support parallel CPU-bound tasks in hierarchy building
    assert args.multi_doc or args.num_workers_cpu is None, "[Error] num_workers_cpu is only supported in multi_doc mode."

    if args.multi_doc:
        print("[Mode] Multi-document memory.")
        metadata, embeddings, doc_mask = load_merged_doc(args.dataset)
        N = embeddings.shape[0]
        if doc_mask.shape != (N, N):
            raise ValueError(f"[multi_doc] doc_mask shape {doc_mask.shape} mismatch with N={N}.")

        cam = CAM(dataset=args.dataset,
                  threshold=args.threshold,
                  weight=args.weight,
                  sigma=args.sigma,
                  top_k=args.k,
                  api_key_path=args.api_key_path,
                  model=args.model,
                  embedding_model=args.embedding_model,
                  max_cluster_size=args.max_cluster_size,
                  summary_field=args.summary_field)

        cam.build_memory(book_title="merged",
                         metadata=metadata,
                         embeddings=embeddings,
                         doc_mask=doc_mask,
                         max_hierarchy_level=args.max_hierarchy_level,
                         num_workers_cpu=args.num_workers_cpu,
                         num_workers_io=args.num_workers_io)
        print("[Done] Multi-document memory completed.")

    else:
        print("[Mode] Single-document memory.")
        chunks_dir = f'./data/{args.dataset}/chunks/'
        files = [f for f in os.listdir(chunks_dir)
                 if f.endswith('.json') and f"_chunked_{args.chunk_size}.json" in f]
        # sample 10 files for testing
        files = random.sample(files, min(args.sample_num, len(files)))
        files.sort()

        if not files:
            print(f"[Warn] No files matched '*_chunked_{args.chunk_size}.json' in {chunks_dir}")
            return

        default_procs = min(10, cpu_count() or 1)
        num_processes = args.num_processes or default_procs
        print(f"[Pool] Using processes: {num_processes}")

        cam_kwargs = dict(
            dataset=args.dataset,
            threshold=args.threshold,
            weight=args.weight,
            sigma=args.sigma,
            top_k=args.k,
            api_key_path=args.api_key_path,
            model=args.model,
            embedding_model=args.embedding_model,
            max_cluster_size=args.max_cluster_size,
            summary_field=args.summary_field
        )

        safe_num_workers_cpu = None
        worker = partial(single_doc_worker,
                         dataset=args.dataset,
                         chunk_size=args.chunk_size,
                         cam_kwargs=cam_kwargs,
                         max_hierarchy_level=args.max_hierarchy_level,
                         num_workers_cpu=safe_num_workers_cpu,
                         num_workers_io=args.num_workers_io,
                         conversation_mode=args.conversation_mode)

        successes = 0
        with Pool(processes=num_processes) as pool:
            for book_title, ok, msg in pool.imap_unordered(worker, files):
                if ok:
                    successes += 1
                    print(f"[✓] {book_title}")
                else:
                    print(f"[✗] {book_title}: {msg}")

        print(f"[Summary] {successes}/{len(files)} single-document memories completed.")


if __name__ == "__main__":
    main()
