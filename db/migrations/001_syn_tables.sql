CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS syn_source_documents (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_key    text NOT NULL,
    tool_name    text NOT NULL,
    concept      text NOT NULL,
    context      text NOT NULL DEFAULT '',
    raw_text     text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz,
    UNIQUE (query_key, tool_name)
);
CREATE INDEX IF NOT EXISTS syn_source_documents_query_key_idx
    ON syn_source_documents (query_key);

CREATE TABLE IF NOT EXISTS syn_chunk_vectors (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_key    text NOT NULL,
    chunk_id     text NOT NULL,
    text         text NOT NULL,
    source_tool  text NOT NULL,
    query        text NOT NULL DEFAULT '',
    "index"      integer NOT NULL,
    embedding    vector(1024) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz,
    UNIQUE (query_key, chunk_id)
);
CREATE INDEX IF NOT EXISTS syn_chunk_vectors_query_key_idx
    ON syn_chunk_vectors (query_key);
CREATE INDEX IF NOT EXISTS syn_chunk_vectors_embedding_hnsw_idx
    ON syn_chunk_vectors USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS syn_synonym_outputs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_key    text NOT NULL UNIQUE,
    concept      text NOT NULL,
    context      text NOT NULL DEFAULT '',
    answer       text NOT NULL,
    progress     text NOT NULL DEFAULT '',
    tools_used   text[] NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz
);
CREATE INDEX IF NOT EXISTS syn_synonym_outputs_created_at_idx
    ON syn_synonym_outputs (created_at DESC);
