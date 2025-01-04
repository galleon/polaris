from qdrant_client import QdrantClient

from settings import settings

client = QdrantClient(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
    index_name=settings.qdrant_index_name,
)