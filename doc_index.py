import os

os.environ["OPENAI_API_BASE"] = "<your_openai_api_base_url>"
os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"

import glob
from minirag import MiniRAG
from minirag.llm import (
    # hf_model_complete,
    gpt_4o_mini_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
DATABASE_PATH = "./knowledge_base"

os.makedirs(os.path.join(DATABASE_PATH, "index"), exist_ok=True)

# Initialize MiniRAG instance
rag = MiniRAG(
    working_dir=os.path.join(DATABASE_PATH, "index"),
    # llm_model_func=hf_model_complete,
    llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=200,
    llm_model_name="gpt-4o-mini",
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)

# Index documents from the specified data path
doc_files = glob.glob(os.path.join(DATABASE_PATH, "documents", "*.txt"))

print("Document files to index:", doc_files)

for idx, doc_file in enumerate(doc_files, 1):
    doc_name = os.path.basename(doc_file)
    print(f"[{idx}/{len(doc_files)}] {doc_name}")

    with open(doc_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    rag.insert(content)

print("Indexing completed.")

for file in os.listdir(os.path.join(DATABASE_PATH, "index")):
    size = os.path.getsize(os.path.join(DATABASE_PATH, "index", file)) / 1024 # kb
    print(f"\t- {file}: {size:.2f} KB")
    