import io
import os
import random
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pypdfium2 as pdfium
from qdrant_client import models
from smolagents import Tool

import lmdb
from polaris.embeddings import Embeddings
from polaris.vectorstore import client as qdrant_client


class IndexerTool(Tool):
    name = "indexer"
    description = "Indexes images from PDFs in a directory into a Qdrant collection"
    inputs = {
        "collection_name": {
            "type": "string",
            "description": "The name of the collection to index the images in",
        },
        "directory_name": {
            "type": "string",
            "description": "The directory containing the PDFs to index",
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qdrant_client = qdrant_client
        self.embeddings = Embeddings()
        self.embeddings.load_model()
        self.lmdb_env = lmdb.open("lmdb", map_size=1024**4)  # 1 TB
        self.namespace = "1234-5678-"

    def extract_scale_from_first_page(self, page, target_font_pixels=24):
        """Extract scale based on the font size of text on the first page."""
        try:
            text_page = page.get_textpage()
            # Iterate over characters to find the first with a defined font size
            font_size = 10_000
            for i in range(text_page.count_chars()):
                font_size = min(font_size, text_page.get_font_size(i))
            if font_size != 10_000 and font_size > 0:
                return target_font_pixels / font_size
        except Exception as e:
            print(f"Failed to extract font size from the first page: {e}")
        # Default to a scale corresponding to 300 DPI
        return 300 / 72

    def embed_images(self, images: list) -> list[list[float]]:
        embeddings = []
        for image in images:
            embeddings.append(self.embeddings.embed_images([image]))
        return embeddings

    def process_pdf(self, collection_name: str, pdf_path: Path):
        pdf = pdfium.PdfDocument(pdf_path)
        print(f"Processing: {str(pdf_path)} [{len(pdf)} pages]")

        # Generate a unique hash for the PDF
        path_hash = str(uuid5(NAMESPACE_URL, str(pdf_path)))

        # Determine scale dynamically from the first page
        first_page = pdf[0]
        scale = self.extract_scale_from_first_page(first_page)
        print(f"Using scale: {scale:.2f}")

        for page_number in range(len(pdf)):
            try:
                page = pdf[page_number]
                image = page.render(scale=scale)

                # Save image as JPEG bytes for storage
                pil_image = image.to_pil()
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="JPEG", quality=85)
                jpg_bytes = img_buffer.getvalue()

                # Save metadata
                img_metadata = {
                    "height": pil_image.height,
                    "width": pil_image.width,
                    "page": page_number + 1,
                    "path": str(pdf_path),
                    "hash": path_hash,
                }

                # Store in LMDB
                with self.lmdb_env.begin(write=True) as txn:
                    key = f"{path_hash}_{page_number + 1:04d}".encode("utf-8")
                    txn.put(key, jpg_bytes)

                # Embed the image and index it
                img_list = [pil_image]
                print("--- ", type(img_list), type(img_list[0]))
                embeddings_list = self.embed_images([pil_image])

                # Index the image
                # self.qdrant_client.upsert(
                #     collection_name=collection_name,
                #     points=[
                #         models.PointStruct(
                #             id=random.getrandbits(64),
                #             payload=img_metadata,
                #             vector=embeddings_list[0],
                #         )
                #     ],
                # )
                print(f"Indexed page {page_number + 1} successfully.")
            except Exception as e:
                print(f"Failed to process page {page_number + 1}: {e}")

    def forward(self, collection_name: str, directory_name: str) -> str:
        # Ensure collection exists
        # get_collections returns a tuple with the first element being the status and
        # the second element being the list of collections
        collections = self.qdrant_client.get_collections()
        print(collections, type(collections))
        collections_list = []
        for collection in collections:
            for c in list(collection[1]):
                collections_list.append(c.name)
        print(collections_list)

        if collection_name not in collections_list:
            print(f"Creating collection: {collection_name}")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                shard_number=4,
                vectors_config={
                    "mean_pooling": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        on_disk=False,
                    ),
                },
            )

        # Process all PDFs in the directory
        for root, _, files in os.walk(directory_name):
            for file in files:
                if file.lower().endswith(".pdf"):
                    self.process_pdf(collection_name, Path(root) / file)

        return f"Indexing completed for collection: {collection_name}"
