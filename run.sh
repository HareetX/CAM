# If already have data/<dataset>/books/ folder with book txt files, skip the transformation.
if [ ! -d "data/LongMemEval/books/" ]; then
    python data/LongMemEval/haystack_sessions2book.py
else
    echo "data/LongMemEval/books/ folder already exists. Skipping book transformation."
fi
# If already have data/<dataset>/chunks/ folder with chunked json files, skip the chunking.
if [ ! -d "data/LongMemEval/chunks/" ]; then
    python prototype/chunks.py --dataset LongMemEval --chunk_size 512
else
    echo "data/LongMemEval/chunks/ folder already exists. Skipping chunking."
fi
# If already have processed_data/<dataset>/ folder with preprocessed embeddings and metadata, skip the preprocessing.
if [ ! -d "processed_data/LongMemEval/" ]; then
    python prototype/preprocess_chunks.py --dataset LongMemEval --model gpt-4o-mini --embedding_model text-embedding-3-large --generate_gist --extract_entity --max_workers 10
else
    echo "processed_data/LongMemEval/ folder already exists. Skipping preprocessing."
fi
# Run constructivist memory model
python prototype/constructivist_memory.py --dataset LongMemEval --chunk_size 512 --threshold 0.6 --weight 0.6 --sigma 1.0 --k 10 --max_cluster_size 12 --max_hierarchy_level 10 --model gpt-4o-mini --embedding_model text-embedding-3-large --num_processes 10 --sample_num 120 --num_workers_io 10