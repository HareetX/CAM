# Constructivist Agentic Memory

<img src="./image/cam_perspective.png" width="55%" alt="CAM Overview"/>

[📄 [Paper](https://arxiv.org/abs/2510.05520)] _CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension_

**CAM** (Constructivist Agentic Memory) is a constructivist-inspired memory framework that improves long-form reading comprehension for LLM-based agents. This is an early-stage release, and we plan to extend CAM with more advanced capabilities in our future work.


## 🛠️ Requirements
- python == 3.9.19
- numpy == 1.26.4
- tqdm == 4.66.1
- datasets == 2.20.0
- networkx == 3.4.2
- openai == 0.28.0
- rouge-score == 0.1.2
- scikit-learn == 1.6.1

## 📚 Datasets

| Dataset | Task | Access |
|----------|------|--------|
| **NovelQA** | Narrative QA | [Request Access](https://github.com/NovelQA/novelqa.github.io) |
| **FABLES** | Claim Verification | [Request Access](https://github.com/mungg/FABLES) |
| **QMSum** | Query-Based Summarization | [GitHub](https://github.com/Yale-LILY/QMSum) |
| **ODSum** | Query-Based Summarization | [GitHub](https://github.com/yale-nlp/ODSum) |
| **MultiHop-RAG** | Multi-Hop Reasoning | [GitHub](https://github.com/yixuantt/MultiHop-RAG) |

[QMSum](https://github.com/Yale-LILY/QMSum), [ODSum](https://github.com/yale-nlp/ODSum), and [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) are publicly available. For [NovelQA](https://github.com/NovelQA/novelqa.github.io) and [FABLES](https://github.com/mungg/FABLES), full documents and ground truth labels are not released to prevent data contamination. Please contact the original authors to request access.

## 🚀 Quick Start

#### 1. Document chunking:
```
python prototype/chunks.py --dataset <dataset_name> --chunk_size <maximum number of tokens per chunk>
```

#### 2. Generate chunk embeddings and extract salient entities (optional) using an LLM:
```
python prototype/preprocess_chunks.py --dataset <dataset_name> --model <LLM model to use> --embedding_model <embedding model to use> --generate_gist --extract_entity
```
This step creates a folder ```./processed_data/``` containing chunk embeddings and metadata.

#### 3. Constructivist memory construction:
```
python prototype/constructivist_memory.py --dataset <dataset_name> --chunk_size <maximum number of tokens per chunk> --threshold <edge activation threshold> --weight <weight for text similarity vs proximity> --sigma <sigma for Gaussian proximity similarity> --k <top-k neighbors per node> --max_cluster_size <maximum nodes allowed in one cluster> --max_hierarchy_level <maximum hierarchy levels> --model <LLM model to use> --embedding_model <embedding model to use>
``` 
This step creates ```./super_graphs/``` (memory structures) and ```./super_embeddings/``` (node embeddings).

#### 4. Prune-and-Grow inference for NovelQA (multichoice setting):
```
python prototype/tasks/question_answering.py --dataset NovelQA --mode MC
```
Outputs are stored in the ```./output/``` directory.

## 📜 Acknowledgement
This project builds on insights from developmental cognitive theory, graph learning, and long-context LLM research. We thank the authors of the benchmark datasets for their valuable contributions.