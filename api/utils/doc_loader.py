"""Shared document loading utilities for RAG pipelines."""

from __future__ import annotations

from pathlib import Path


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def load_documents(file_path: str):
    """Load documents from a file path using the appropriate LangChain loader."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()
    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path).load()
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path).load()
    else:
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8").load()
