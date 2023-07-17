# Chromadb settings
import os
from chromadb.config import Settings

CHROMADB_SETTING = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory="db", anonymized_telemetry=False
)
