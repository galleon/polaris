from qdrant_client import QdrantClient

from .settings import settings

client = QdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    api_key=settings.QDRANT_TOKEN,
    # index_name=settings.QDRANT_INDEX_NAME,
)
