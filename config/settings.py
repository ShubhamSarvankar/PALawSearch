from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Elasticsearch — es_password has no default so a missing env var fails loudly
    es_host: str = "https://localhost:9200"
    es_user: str = "elastic"
    es_password: str
    es_index_bm25: str = "pa_cases_bm25"
    es_index_dense: str = "pa_cases_dense"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Models — paths updated to trained checkpoints in Phase 3
    encoder_model_path: str = "nlpaueb/legal-bert-base-uncased"
    cross_encoder_model_path: str = "BAAI/bge-reranker-large"
    dense_vector_dim: int = 768

    # Retrieval — reconciled to 200 (architecture §10 #7)
    top_k_rerank: int = 200

    # Ollama RAG generation
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    # Phase 4 LLM judge — Anthropic API key, no default so missing .env fails loudly
    anthropic_api_key: str

    # Flask API
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_debug: bool = True


settings = Settings()
