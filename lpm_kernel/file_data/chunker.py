from typing import List
from lpm_kernel.L1.bio import Chunk
import traceback
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()


class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

    def split(self, content: str) -> List[Chunk]:
        try:
            if not content:
                logger.warning("Empty content provided")
                return []

            logger.info(f"Starting to split content of length {len(content)}")

            # use LangChain splitter
            texts = self.text_splitter.split_text(content)

            chunks = [
                Chunk(
                    id=None,
                    document_id=None,
                    content=text,
                    embedding=None,
                    tags=None,
                    topic=None,
                )
                for text in texts
            ]

            logger.info(f"Split completed, created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error in split method: {str(e)}")
            logger.error(traceback.format_exc())
            raise
