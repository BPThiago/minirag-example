import os

os.environ["OPENAI_API_BASE"] = "<your_openai_api_base_url>"
os.environ["OPENAI_API_KEY"] = "<your_openai_api_key>"

from minirag import MiniRAG, QueryParam
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

queries = [
    {
        "query": "Quem é o CEO da TechFlow?",
        "gold_answer": "João Silva",
        "tipo": "single-hop", # informação direta
        "documentos_fonte": ["product_launch.txt", "team_meeting.txt", "investor_update.txt"]
    },
    {
        "query": "Quando foi lançado o produto Alpha?",
        "gold_answer": "15 de janeiro de 2024",
        "tipo": "single-hop", # data específica
        "documentos_fonte": ["product_launch.txt"]
    },
    {
        "query": "Quem lidera a equipe de engenharia da TechFlow?",
        "gold_answer": "Sarah Chen (CTO)",
        "tipo": "single-hop", # cargo e pessoa
        "documentos_fonte": ["product_launch.txt", "team_meeting.txt"]
    },
    {
        "query": "Quantos usuários simultâneos o Alpha suportou durante o lançamento?",
        "gold_answer": "500 mil usuários simultâneos",
        "tipo": "single-hop", # métrica específica
        "documentos_fonte": ["team_meeting.txt"]
    },
    {
        "query": "Qual foi a taxa de conversão na primeira semana e ela atingiu a meta?",
        "gold_answer": "8%, sim, superou a meta de 5%",
        "tipo": "multi-hop", # combinação de diferentes pontos
        "documentos_fonte": ["team_meeting.txt"]
    },
    {
        "query": "Quais foram os principais desafios técnicos no lançamento do Alpha?",
        "gold_answer": "Escalabilidade de banco de dados (migração PostgreSQL para híbrido com Redis), performance do dashboard (redução de latência de 2s para 200ms com WebSockets), e monitoramento 24/7 (implementação Prometheus/Grafana)",
        "tipo": "multi-hop", # sumarização de múltiplos pontos
        "documentos_fonte": ["tech_challenges.txt"]
    },
    {
        "query": "Quantos desenvolvedores tem a equipe técnica atualmente?",
        "gold_answer": "15 desenvolvedores (expandida de 8 para 15)",
        "tipo": "multi-hop", # requer cronologia
        "documentos_fonte": ["team_meeting.txt", "tech_challenges.txt", "investor_update.txt"]
    },
    {
        "query": "Qual é o runway e burn rate mensal da empresa?",
        "gold_answer": "Runway de 18 meses e burn rate de R$ 150.000 mensais",
        "tipo": "multi-hop", # combina dois dados financeiros
        "documentos_fonte": ["investor_update.txt"]
    },
    {
        "query": "Sarah Chen recebeu bônus? Se sim, qual valor?",
        "gold_answer": "Sim, R$ 5.000 (aprovado por João Silva para equipes técnica e marketing)",
        "tipo": "multi-hop", # condicional + valor + contexto
        "documentos_fonte": ["team_meeting.txt"]
    },
    {
        "query": "Resuma o desempenho comercial do Alpha até fevereiro de 2024",
        "gold_answer": "10.000 usuários cadastrados, taxa de conversão de 8% (acima da meta), pipeline de R$ 2 milhões para Q1, MRR de R$ 80.000, e menções positivas em 15 publicações",
        "tipo": "sumarização",
        "documentos_fonte": ["team_meeting.txt", "investor_update.txt"]
    }
]

os.makedirs("./logs", exist_ok=True)

for idx, q in enumerate(queries, 1):
    print(f"Query [{idx}/{len(queries)}]: {q['query']}")

    print("[1] Context")
    context = rag.query(
        q["query"],
        param=QueryParam(
            mode="mini",
            only_need_context=True,
            response_type="Concise Answer", # Aqui não importa muito
        )
    )

    # print(f"{context[:200]}...\n") # Inclui entidades e documentos fontes
    entities_section = context.split("-----Entities-----")[1].split("-----Sources-----")[0]
    
    # aproximado! precisaria de uma parse mais honesto
    print(f"✓ Entidades: {entities_section[:200].strip()}...")
    num_entities = len([line for line in entities_section.splitlines() if line.startswith('"""') and line.endswith('"""')])
    print("Number of entities retrieved:", num_entities)
    
    sources_section = context.split("-----Sources-----")[1]
    print(f"✓ Documentos fontes: {sources_section.strip()}\n")

    print("[2] Answer")

    minirag_answer = rag.query(
        q["query"], 
        param=QueryParam(
            mode="mini", 
            response_type="Concise Answer"
        )
    )
    print("Answer:\n", minirag_answer)
    print("Gold Answer:", q["gold_answer"])
    print()
    
    with open(f"./logs/query_{idx}.txt", mode="w", encoding="utf-8") as log_file:
        log_file.write("Query:\n")
        log_file.write(q["query"] + "\n\n")
        log_file.write("Context:\n")
        log_file.write(context + "\n\n")
        log_file.write("Answer:\n")
        log_file.write(minirag_answer + "\n\n")
        log_file.write("Gold Answer:\n")
        log_file.write(q["gold_answer"] + "\n")