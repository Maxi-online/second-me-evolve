# Embedding Model Switching Guide

## Understanding Embedding Dimensions

When using different embedding models (like switching from OpenAI to Ollama models), you may encounter dimension mismatch issues. This happens because different models produce embedding vectors with different dimensions:

| Model | Dimension |
|-------|----------|
| OpenAI text-embedding-ada-002 | 1536 |
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| Ollama snowflake-arctic-embed | 768 |
| Ollama nomic-embed-text | 768 |
| Ollama mxbai-embed-large | 1024 |

## Handling Dimension Mismatches

Second Me now includes automatic detection and handling of embedding dimension mismatches. When you switch between embedding models with different dimensions, the system will:

1. Detect the dimension of the new embedding model
2. Check if the existing ChromaDB collections have a different dimension
3. If a mismatch is detected, automatically reinitialize the collections with the new dimension
4. Provide clear error messages and logging information about the process

## Recommended Workflow for Switching Models

When switching between embedding models with different dimensions, follow these steps:

1. Update your embedding model configuration in Settings
2. Restart the application to ensure proper initialization
3. If you encounter any issues, you can manually reset the vector database:
   - Delete the contents of the `data/chroma_db` directory
   - Restart the application

## Troubleshooting

The system now automatically handles dimension mismatches when switching between embedding models. You'll see log messages like:

```
Warning: Existing 'documents' collection has dimension X, but current model requires Y
Automatically reinitializing ChromaDB collections with the new dimension...
Successfully reinitialized ChromaDB collections with the new dimension
```

This indicates that the system has detected and resolved a dimension mismatch automatically. If you still encounter issues after the automatic handling:

1. Check the application logs for any error messages
2. If problems persist, you can manually reset the vector database:
   - Stop the application
   - Delete the contents of the `data/chroma_db` directory
   - Restart the application

## Technical Details

The dimension mismatch handling is implemented in:

- `lpm_kernel/file_data/chroma_utils.py`: Contains utilities for detecting model dimensions and reinitializing collections
- `lpm_kernel/file_data/embedding_service.py`: Handles dimension checking during initialization
- `docker/app/init_chroma.py`: Performs dimension validation during initial setup

The system maintains a mapping of known embedding models to their dimensions and will default to 1536 (OpenAI's dimension) for unknown models.