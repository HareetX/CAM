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
from tasks.tools.prompts import text_summarization_template, kg_community_summarization_template
from itertools import combinations
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from cdlib import algorithms

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

    def _kg_community_summarization(self, community: set, cluster_type: str = "node"):
        """Summarize a KG community into a text gist."""
        assert cluster_type in ["node", "edge"], "Unsupported cluster type for KG community summarization."
        # Prepare input
        entities = []
        relationships = []

        if cluster_type == "node": # node cluster
            for node_id in community:
                node_data = self.kg.nodes[node_id]
                entities.append((node_id, node_data.get("entity_name"), node_data.get("entity_description")))
                for neighbor in self.kg.neighbors(node_id):
                    # Not only within community
                    edge_data = self.kg.get_edge_data(node_id, neighbor)
                    relationships.append((edge_data.get("source_entity"),
                                          edge_data.get("target_entity"),
                                          edge_data.get("relationship_description")))
        else:  # edge cluster
            ent_set = set()
            for edge in community:
                n_a, n_b = edge
                edge_data = self.kg.get_edge_data(n_a, n_b)
                relationships.append((edge_data.get("source_entity"),
                                      edge_data.get("target_entity"),
                                      edge_data.get("relationship_description")))
                ent_set.add(n_a)
                ent_set.add(n_b)
            for node_id in ent_set:
                node_data = self.kg.nodes[node_id]
                entities.append((node_id, node_data.get("entity_name"), node_data.get("entity_description")))

        input_texts = ""
        input_texts += "Entities:\n"
        input_texts += "entity_name, entity_description\n"
        for eid, name, desc in entities:
            input_texts += f"{name}, {desc}\n"

        input_texts += "\nRelationships:\n"
        input_texts += "source_entity, target_entity, relationship_description\n"
        for src, tgt, desc in relationships:
            input_texts += f"{src}, {tgt}, {desc}\n"

        # Summarization prompt
        prompt = kg_community_summarization_template.format(input_texts=input_texts)

        response = self.client.obtain_response(prompt, max_tokens=1024, temperature=0.0)

        super_gist = response.strip()
        super_emb = np.array(self.client.obtain_embedding(super_gist), dtype=np.float32)

        return super_gist, super_emb

    def _add_nodes_from_kg_leiden(self, metadata: list[dict], chunk_embedding: np.ndarray,  title_for_default_doc: str, resolution_parameter: float = 1.0):
        """
        Add nodes from constructed KG level.
        """
        # Use Leiden communities to detect 0-level nodes for each chunk
        N = 0
        chunk_community_map = {} # chunk_id -> list of communities (sets of node ids)
        for chunk_id in range(len(metadata)):
            # Extract subgraph for this chunk
            edges_in_chunk = [ (u, v) for u, v, data in self.kg.edges(data=True) if chunk_id == data.get("chunk_id", -1) ]
            subgraph = self.kg.edge_subgraph(edges_in_chunk)

            if subgraph.number_of_nodes() == 0: # No edges in this chunk
                nodes_in_chunk = [n for n, data in self.kg.nodes(data=True) if chunk_id in data.get("chunk_id")]
                communities = [set(nodes_in_chunk)]
                N += len(communities)
                chunk_community_map[chunk_id] = communities # len is 1
                continue

            # Clustering
            coms = algorithms.leiden(subgraph, weights="relationship_strength")
            communities = list(coms.communities)

            N += len(communities)

            chunk_community_map[chunk_id] = communities

        self.node_ids = list(range(N))
        self.doc_ids = []

        i = 0
        embeddings = []
        node_communities = [] # list of sets of node ids
        for chunk_id, communities in chunk_community_map.items():
            doc_id = title_for_default_doc
            self.doc_ids.append(doc_id)

            if len(communities) == 1: # Let the whole chunk be one community
                comm = communities[0]

                entity_concepts = metadata[chunk_id].get("entity_concepts", [])
                text = metadata[chunk_id].get("text", "")
                gist = metadata[chunk_id].get("gist", "")
                emb = chunk_embedding[chunk_id]

                embeddings.append(emb)

                self.memory.add_node(
                    i,
                    chunk_id=chunk_id,
                    text=text,
                    gist=gist,
                    entities=entity_concepts,
                    doc_id=doc_id
                )

                # Add edges between communities if there are shared entities
                for prev_i in range(i):
                    prev_comm = node_communities[prev_i]
                    overlap_count = len(set(comm) & set(prev_comm))
                    if overlap_count > 0:
                        self.memory.add_edge(i, prev_i, weight=1.0)

                node_communities.append(comm)

                i += 1

                continue

            for comm in communities:
                entity_concepts = [self.kg.nodes[n].get("entity_name") for n in comm]
                # Summarize node text & gist from member entities & relationships
                text, emb = self._kg_community_summarization(comm, cluster_type="node")

                embeddings.append(emb)

                self.memory.add_node(
                    i,
                    chunk_id=chunk_id,
                    text=text,
                    gist=text,
                    entities=entity_concepts,
                    doc_id=doc_id
                )

                # Add edges between communities if there are shared entities
                for prev_i in range(i):
                    prev_comm = node_communities[prev_i]
                    overlap_count = len(set(comm) & set(prev_comm))
                    if overlap_count > 0:
                        self.memory.add_edge(i, prev_i, weight=1.0)

                # Add edges between communities if there are relationships between their member nodes
                for prev_i in range(i):
                    prev_comm = node_communities[prev_i]
                    edge_weight = 0.0
                    count = 0
                    for n1 in comm:
                        for n2 in prev_comm:
                            if self.kg.has_edge(n1, n2):
                                edge_data = self.kg.get_edge_data(n1, n2)
                                edge_weight += edge_data.get("relationship_strength", 0.0)
                                count += 1
                    if count > 0:
                        edge_weight /= count
                    if edge_weight > 0.0:
                        self.memory.add_edge(i, prev_i, weight=float(edge_weight))

                node_communities.append(comm)

                i += 1
        return np.stack(embeddings, axis=0)

    def _add_kg_level_graph_from_kg(self, num_workers_io: int = None):
        self.kg_level = nx.Graph()
        self.kg_level_embeddings = []

        # Both entities and relationships are the nodes in this level, the edges are links between them
        node_id = 0
        node_map = {}  # original node id -> kg_level node id
        edge_map = {}  # original edge (u,v) -> kg_level node id

        for n, data in self.kg.nodes(data=True):
            text = f"{data.get('entity_name')}: {data.get('entity_description')}"

            self.kg_level.add_node(
                node_id,
                text=text,
                node_type="entity",
                entity_name=data.get("entity_name"),
                entity_description=data.get("entity_description"),
                chunk_id=data.get("chunk_id"),
            )

            node_map[n] = node_id

            node_id += 1

        for u, v, data in self.kg.edges(data=True):
            text = data.get("relationship_description", "")

            self.kg_level.add_node(
                node_id,
                text=text,
                node_type="relationship",
                source_entity=data.get("source_entity"),
                target_entity=data.get("target_entity"),
                relationship_description=data.get("relationship_description"),
                relationship_strength=data.get("relationship_strength"),
                chunk_id=[data.get("chunk_id")],
            )

            edge_map[(u, v)] = node_id

            node_id += 1

        # Now add edges between entity nodes and relationship nodes
        for u, v, data in self.kg.edges(data=True):
            rel_node_id = edge_map[(u, v)]
            src_entity_node_id = node_map[u]
            tgt_entity_node_id = node_map[v]

            self.kg_level.add_edge(rel_node_id, src_entity_node_id, weight=1.0)
            self.kg_level.add_edge(rel_node_id, tgt_entity_node_id, weight=1.0)

        # Request embeddings for kg_level nodes in parallel
        print(f"[KG Level] Obtaining embeddings in {num_workers_io} threads..." if num_workers_io is not None else "[KG Level] Obtaining embeddings in single thread...")
        if num_workers_io is None:
            for n, data in self.kg_level.nodes(data=True):
                text = data.get("text", "")
                emb = np.array(self.client.obtain_embedding(text), dtype=np.float32)
                self.kg_level_embeddings.append(emb)
        else:
            texts = [data.get("text", "") for n, data in self.kg_level.nodes(data=True)]
            with ThreadPool(num_workers_io) as pool:
                func = partial(self.client.obtain_embedding)
                embeddings = pool.map(func, texts)
            for emb in embeddings:
                emb_array = np.array(emb, dtype=np.float32)
                self.kg_level_embeddings.append(emb_array)

        assert len(self.kg_level_embeddings) == self.kg_level.number_of_nodes(), "Mismatch in KG level embeddings and nodes."

        self.kg_level_embeddings = np.stack(self.kg_level_embeddings, axis=0)

        return node_map, edge_map

    def _add_nodes_from_kg_hierarchical_link_community(self, metadata: list[dict], chunk_embedding: np.ndarray, title_for_default_doc: str, conversation_mode: str = "kg_edge_cluster_wo_kg", num_workers_io: int = None):
        """
        Add nodes from constructed KG level.
        """
        if conversation_mode == "kg_edge_cluster_w_kg":
            # Add KG level graph to self.kg_level, embeddings to self.kg_level_embeddings
            node_map, edge_map = self._add_kg_level_graph_from_kg(num_workers_io=num_workers_io)
            print(f"[KG Level] KG level graph constructed with {self.kg_level.number_of_nodes()} nodes and {self.kg_level.number_of_edges()} edges.")

        # Use hierarchical link communities to detect 0-level nodes for each chunk
        N = 0
        chunk_community_map = {} # chunk_id -> list of communities (sets of edge ids)
        for chunk_id in range(len(metadata)):
            # Extract subgraph for this chunk
            edges_in_chunk = [ (u, v) for u, v, data in self.kg.edges(data=True) if chunk_id == data.get("chunk_id", -1) ]
            subgraph = self.kg.edge_subgraph(edges_in_chunk)

            if subgraph.number_of_nodes() == 0: # No edges in this chunk
                nodes_in_chunk = [n for n, data in self.kg.nodes(data=True) if chunk_id in data.get("chunk_id")]
                communities = [set( (n_a, n_b) for n_a in nodes_in_chunk for n_b in nodes_in_chunk if n_a != n_b )]
                N += len(communities)
                chunk_community_map[chunk_id] = communities # len is 1
                continue

            # Clustering
            coms = algorithms.hierarchical_link_community(subgraph)
            communities = list(coms.communities)

            N += len(communities)

            chunk_community_map[chunk_id] = communities

        self.node_ids = list(range(N))
        self.doc_ids = []

        i = 0
        embeddings = []
        node_communities = [] # list of sets of node ids
        for chunk_id, communities in chunk_community_map.items():
            doc_id = title_for_default_doc
            self.doc_ids.append(doc_id)

            if len(communities) == 1: # Let the whole chunk be one community
                comm = communities[0]

                entity_concepts = set()
                for n_a, n_b in comm:
                    entity_concepts.add(self.kg.nodes[n_a].get("entity_name"))
                    entity_concepts.add(self.kg.nodes[n_b].get("entity_name"))
                entity_concepts = list(entity_concepts)

                text = metadata[chunk_id].get("text", "")
                gist = metadata[chunk_id].get("gist", "")
                emb = chunk_embedding[chunk_id]

                embeddings.append(emb)

                if conversation_mode == "kg_edge_cluster_w_kg":
                    community = set()
                    for n_a, n_b in comm:
                        community.add(node_map[n_a])
                        community.add(node_map[n_b])
                        if self.kg.has_edge(n_a, n_b) and self.kg.edges[n_a, n_b].get("chunk_id") == chunk_id:
                            community.add(edge_map[(n_a, n_b)])
                    community = list(community)

                    self.memory.add_node(
                        i,
                        chunk_id=chunk_id,
                        text=text,
                        gist=gist,
                        entities=entity_concepts,
                        doc_id=doc_id,
                        community=community
                    )
                elif conversation_mode == "kg_edge_cluster_wo_kg":
                    self.memory.add_node(
                        i,
                        chunk_id=chunk_id,
                        text=text,
                        gist=gist,
                        entities=entity_concepts,
                        doc_id=doc_id
                    )
                else:
                    raise ValueError(f"[Error] Unsupported conversation mode {conversation_mode} for KG edge cluster.")

                # Add edges between communities if there are shared entities
                ent_id_set = set()
                for n_a, n_b in comm:
                    ent_id_set.add(n_a)
                    ent_id_set.add(n_b)
                for prev_i in range(i):
                    prev_comm = node_communities[prev_i]
                    # Calculate the overlap count between ent_id_set and prev_comm
                    overlap_count = len(ent_id_set & prev_comm)
                    if overlap_count > 0:
                        self.memory.add_edge(i, prev_i, weight=1.0)

                node_communities.append(ent_id_set)

                i += 1

                continue

            for comm in communities:
                entity_concepts = set()
                for n_a, n_b in comm:
                    entity_concepts.add(self.kg.nodes[n_a].get("entity_name"))
                    entity_concepts.add(self.kg.nodes[n_b].get("entity_name"))
                entity_concepts = list(entity_concepts)
                # Summarize node text & gist from member entities & relationships
                text, emb = self._kg_community_summarization(comm, cluster_type="edge")

                embeddings.append(emb)

                if conversation_mode == "kg_edge_cluster_w_kg":
                    community = set()
                    for n_a, n_b in comm:
                        community.add(node_map[n_a])
                        community.add(node_map[n_b])
                        if self.kg.has_edge(n_a, n_b) and self.kg.edges[n_a, n_b].get("chunk_id") == chunk_id:
                            community.add(edge_map[(n_a, n_b)])
                    community = list(community)

                    self.memory.add_node(
                        i,
                        chunk_id=chunk_id,
                        text=text,
                        gist=text,
                        entities=entity_concepts,
                        doc_id=doc_id,
                        community=community
                    )
                elif conversation_mode == "kg_edge_cluster_wo_kg":
                    self.memory.add_node(
                        i,
                        chunk_id=chunk_id,
                        text=text,
                        gist=text,
                        entities=entity_concepts,
                        doc_id=doc_id
                    )
                else:
                    raise ValueError(f"[Error] Unsupported conversation mode {conversation_mode} for KG edge cluster.")

                # Add edges between communities if there are relationships between their member nodes
                ent_id_set = set()
                for n_a, n_b in comm:
                    ent_id_set.add(n_a)
                    ent_id_set.add(n_b)
                for prev_i in range(i):
                    prev_comm = node_communities[prev_i]
                    # Calculate the overlap count between ent_id_set and prev_comm
                    overlap_count = len(ent_id_set & prev_comm)
                    if overlap_count > 0:
                        self.memory.add_edge(i, prev_i, weight=1.0)

                node_communities.append(ent_id_set)

                i += 1
        return np.stack(embeddings, axis=0)

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
        if isinstance(self.memory.nodes[self.node_ids[0]]['chunk_id'], list):
            # if chunk_id is list, take the min distance
            chunk_ids_list = [self.memory.nodes[n]['chunk_id'] for n in self.node_ids]
            diff = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    dists = [abs(c1 - c2) for c1 in chunk_ids_list[i] for c2 in chunk_ids_list[j]]
                    diff[i, j] = min(dists) if dists else np.inf
        else:
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

    def run_hierarchy(self, book_title, max_hierarchy_level: int = 10, num_workers_cpu: int = None, num_workers_io: int = None, conversation_mode: str = "null"):

        self.level_graphs = []
        self.level_embeddings = []

        start_level = 0

        print(f"[Hierarchy] Building hierarchy for {book_title} on {conversation_mode} mode...")
        if conversation_mode == "kg_edge_cluster_w_kg":
            print(f"[Hierarchy] Starting from KG level for {book_title}...")
            kg_level_community_dict = {}
            kg_level_overlap_communities = []
            for n in self.kg_level.nodes():
                for comm_id, comm_nodes in self.memory.nodes(data=True):
                    kg_level_overlap_communities.append(set(comm_nodes.get("community", [])))
                    if n in comm_nodes.get("community", []):
                        kg_level_community_dict.setdefault(n, []).append(comm_id)

            self._print_community_stats(0, book_title, self.kg_level, kg_level_overlap_communities)
            self._save_graph_with_community(self.kg_level, kg_level_community_dict, 0)
            self._save_level_graph_and_embeddings(graph=self.kg_level, embeddings=self.kg_level_embeddings, level=0)

            start_level = 1

        # Level 0/1: current memory state
        self._save_level_graph_and_embeddings(graph=self.memory, embeddings=self.embeddings, level=start_level)

        current_graph = self.memory.copy()
        prev_num_communities = current_graph.number_of_nodes()
        level = start_level

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
        self.run_hierarchy(book_title, max_hierarchy_level=max_hierarchy_level, num_workers_cpu=num_workers_cpu, num_workers_io=num_workers_io, conversation_mode=conversation_mode)

    def _entity_similarity(self, ent_a_type: list[str], ent_a_emb: np.array, ent_b_type: list[str], ent_b_emb: np.array, alpha_meta: float) -> float:
        """Similarity between two entity concepts using embeddings."""
        # Get embeddings
        emb_a = np.array(ent_a_emb, dtype=np.float32)
        emb_b = np.array(ent_b_emb, dtype=np.float32)

        # Compute cosine similarity
        cos_sim = cosine_similarity(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0][0]

        # Entity type bonus (entity_type is list[str])
        overlap_types = set(ent_a_type) & set(ent_b_type)
        min_type_num = min(len(ent_a_type), len(ent_b_type))
        type_bonus = len(overlap_types) / min_type_num if min_type_num > 0 else 0.0

        final_sim = alpha_meta * cos_sim + (1 - alpha_meta) * type_bonus
        return final_sim

    def _construct_kg(self, kg_metadata: list[dict]):
        kg = nx.Graph()

        ent_embs = []
        ent_list = []

        ent_id_map = {} # (chunk_id, entity_name) -> ent_id
        ent_id = 0

        for kg_data in kg_metadata:
            # print(f"[KG] Processing chunk_id={kg_data['chunk_id']}...")
            chunk_id = kg_data['chunk_id']
            graph_elements = kg_data.get('graph_elements', {"entities": [], "relationships": []})
            entities = graph_elements.get('entities', [])
            relations = graph_elements.get('relationships', [])

            for ent in entities:
                key = (chunk_id, ent['entity_name'])
                if key not in ent_id_map:
                    # Entity linking (Add new node or Link to existing)
                    emb = np.array(self.client.obtain_embedding(f"{ent['entity_name']}: {ent['entity_description']}"), dtype=np.float32)
                    ent_name_lower = ent['entity_name'].lower()
                    ent_type_lower = [t.lower() for t in ent.get('entity_type', [])]
                    linked_id = -1
                    sim_max = -1.0
                    for existing_ent_id, existing_ent in enumerate(ent_list):
                        existing_name_lower = existing_ent['entity_name'].lower()
                        existing_type_lower = [t.lower() for t in existing_ent.get('entity_type', [])]
                        existing_emb = ent_embs[existing_ent_id]

                        if existing_name_lower == ent_name_lower:
                            sim_max = 1.0
                            linked_id = existing_ent_id
                            break

                        sim = self._entity_similarity(ent_type_lower, emb, existing_type_lower, existing_emb, alpha_meta=0.7)

                        if sim > sim_max:
                            sim_max = sim
                            linked_id = existing_ent_id

                    if linked_id != -1 and sim_max >= 0.85:
                        # Link to existing entity
                        # print(f"[KG] Linking entity '{ent['entity_name']}' in chunk {chunk_id} to existing entity id {linked_id} '{ent_list[linked_id]['entity_name']}' (sim={sim_max:.4f}).")
                        kg.nodes[linked_id]['chunk_id'].append(chunk_id)
                        ent_id_map[key] = linked_id
                        # print(f"[KG] Updated node {linked_id} chunk_id list: {kg.nodes[linked_id]['chunk_id']}.")
                        # print(f"[KG] After linking, ent_list size: {len(ent_list)}. node number: {kg.number_of_nodes()}.")
                    else:
                        linked_id = -1
                        ent_id_map[key] = ent_id

                    if linked_id == -1:
                        # Add new entity node
                        # print(f"[KG] Adding new entity '{ent['entity_name']}' in chunk {chunk_id} as entity id {ent_id}.")
                        kg.add_node(ent_id,
                                          chunk_id=[chunk_id],
                                          entity_name=ent['entity_name'],
                                          entity_description=ent['entity_description'],
                                          entity_type=ent.get('entity_type', []))
                        ent_embs.append(emb)
                        ent_list.append(ent)
                        # print(f"[KG] ent_list size: {len(ent_list)}. node number: {kg.number_of_nodes()}.")
                        ent_id += 1

            for rel in relations:
                source_key = (chunk_id, rel['source_entity'])
                target_key = (chunk_id, rel['target_entity'])
                rel_description = rel.get('relationship_description', '')
                rel_strength = rel.get('relationship_strength', 0.0)
                safe_rel_strength = min(max(rel_strength / 10, 0.0), 1.0)
                # print(f"[KG] Processing relation: {rel['source_entity']} --({rel_description}, strength={safe_rel_strength:.4f})--> {rel['target_entity']}")
                if source_key in ent_id_map and target_key in ent_id_map:
                    # print(f"[KG] Adding edge between '{rel['source_entity']}' and '{rel['target_entity']}' with strength {safe_rel_strength:.4f}.")
                    source_id = ent_id_map[source_key]
                    target_id = ent_id_map[target_key]
                    kg.add_edge(source_id, target_id,
                                      source_entity=rel['source_entity'],
                                      target_entity=rel['target_entity'],
                                      relationship_description=rel_description,
                                      relationship_strength=safe_rel_strength,
                                      chunk_id=chunk_id)
                    # print(f"[KG] Edge added between node {source_id} and node {target_id}.")

        # assert 0, "Debug stop"
        return kg

    def build_memory_kg(self,
                        book_title,
                        metadata,
                        embeddings,
                        kg_metadata,
                        doc_mask,
                        max_hierarchy_level: int = 10,
                        num_workers_cpu: int = None,
                        num_workers_io: int = None,
                        conversation_mode: str = 'kg'):

        self._prepare_output_dirs(book_title)

        # Add nodes
        if conversation_mode is not None:
            # Construct KG level
            print(f"[KG] Constructing knowledge graph level for {book_title}...")
            self.kg = self._construct_kg(kg_metadata)
            print(f"[KG] KG constructed for {book_title}: nodes={self.kg.number_of_nodes()}, edges={self.kg.number_of_edges()}.")
            if conversation_mode in ['kg_node_cluster_wo_kg']:
                kg_community_embeddings = self._add_nodes_from_kg_leiden(metadata, chunk_embedding=embeddings, title_for_default_doc=book_title)
                self.embeddings = np.atleast_2d(kg_community_embeddings).astype(np.float32)
            elif conversation_mode in ['kg_edge_cluster_wo_kg', 'kg_edge_cluster_w_kg']:
                kg_community_embeddings = self._add_nodes_from_kg_hierarchical_link_community(metadata, chunk_embedding=embeddings, title_for_default_doc=book_title, conversation_mode=conversation_mode, num_workers_io=num_workers_io)
                self.embeddings = np.atleast_2d(kg_community_embeddings).astype(np.float32)
            elif conversation_mode in ['kg']:
                node_map, edge_map = self._add_kg_level_graph_from_kg(num_workers_io=num_workers_io)
                self.memory = self.kg_level.copy()
                self.embeddings = np.atleast_2d(self.kg_level_embeddings.copy()).astype(np.float32)
                self.node_ids = list(range(self.memory.number_of_nodes()))
                self.doc_ids = [self.memory.nodes[n].get("doc_id") for n in self.node_ids] # TODO: Need to ensure in future
            else:
                raise ValueError(f"[Error] Unknown conversation_mode for KG: {conversation_mode}")
            print(f"[KG] KG communities for {book_title} added as level-0/1 nodes: total nodes={len(self.node_ids)}.")

        else:
            self._add_nodes_from_metadata(metadata, title_for_default_doc=book_title)
            self.embeddings = np.atleast_2d(embeddings).astype(np.float32)
        self.doc_mask = doc_mask  # None for single-doc; (N,N) for multi-doc

        # Build edges from pairwise similarity for level-0
        print(f"[KG] Building pairwise similarity for level-0/1 of {book_title}...")
        sim = self._build_pairwise_similarity()
        print(f"[KG] Adding edges from similarity for level-0/1 of {book_title}...")
        self._add_edges_from_similarity(sim)

        # Run hierarchy & compose all levels
        self.run_hierarchy(book_title, max_hierarchy_level=max_hierarchy_level, num_workers_cpu=num_workers_cpu, num_workers_io=num_workers_io, conversation_mode=conversation_mode)


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

def load_kg(dataset: str, chunk_file_stem: str):
    kg_meta_fp = f'./processed_data/{dataset}/chunk_kg/metadata/{chunk_file_stem}_kg.json'
    kg_meta = load_json(kg_meta_fp)
    return kg_meta

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
    kg_modes = ['kg', 'kg_node_cluster_wo_kg', 'kg_edge_cluster_wo_kg', 'kg_edge_cluster_w_kg']
    if conversation_mode in stmt_modes + kg_modes:
        safe_conversation_mode = conversation_mode

    try:
        # filename: <book>_chunked_<chunk_size>.json
        title_stem = os.path.splitext(filename)[0]  # without .json
        book_title = title_stem.split(f"_chunked_{chunk_size}")[0]

        metadata, embeddings = load_single_doc(dataset, title_stem)
        if safe_conversation_mode is not None and safe_conversation_mode in stmt_modes:
            # load statement-level metadata & embeddings
            stmt_meta, stmt_embs = load_statements(dataset, title_stem)
        if safe_conversation_mode is not None and safe_conversation_mode in kg_modes:
            # load knowledge graph metadata
            kg_meta = load_kg(dataset, title_stem)

        cam = CAM(**cam_kwargs)
        if safe_conversation_mode is None:
            cam.build_memory(book_title=book_title,
                             metadata=metadata,
                             embeddings=embeddings,
                             doc_mask=None,
                             max_hierarchy_level=max_hierarchy_level,
                             num_workers_cpu=num_workers_cpu,
                             num_workers_io=num_workers_io)
        elif safe_conversation_mode in stmt_modes:
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
        elif safe_conversation_mode in kg_modes:
            cam.build_memory_kg(book_title=book_title,
                                metadata=metadata,
                                embeddings=embeddings,
                                kg_metadata=kg_meta,
                                doc_mask=None,
                                max_hierarchy_level=max_hierarchy_level,
                                num_workers_cpu=num_workers_cpu,
                                num_workers_io=num_workers_io,
                                conversation_mode=safe_conversation_mode)
        else:
            raise ValueError(f"[Error] Unknown conversation_mode: {safe_conversation_mode}")

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
