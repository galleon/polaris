from typing import Any, cast

from PIL import Image
from transformers import PreTrainedModel

from polaris.settings import settings


class Embeddings:
    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        self.model_name = model_name
        self.model: PreTrainedModel
        self.processor: Any
        self.mock_image = self.create_mock_image()

    def load_model(self):
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor

        # Default to CPU
        device = torch.device("cpu")
        torch_type = torch.float32

        # Check for MPS support on macOS
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            torch_type = torch.float32  # MPS currently supports only float32

        # Check for CUDA availability
        elif torch.cuda.is_available():
            try:
                available_memory = torch.cuda.mem_get_info()[1]
                if available_memory >= 8 * 1024**3:  # 8 GB
                    device = torch.device("cuda")
                    torch_type = torch.bfloat16
            except RuntimeError:
                pass

        # Model initialization
        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=torch_type,
            device_map=device,
            token=settings.HF_TOKEN,
        ).eval()

        # Processor initialization
        self.processor = cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(self.model_name),
        )

    def create_mock_image(self):
        import numpy as np

        """Creates a blank 448x448 RGB image."""
        return 255 * np.ones((448, 448, 3), dtype=np.uint8)

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        import torch
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            create_stringlist_dataset_class(queries),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.process_queries(x),
        )

        query_embeddings: list[torch.Tensor] = []

        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embedding_query = self.model(**batch_query)
            query_embeddings.extend(torch.unbind(embedding_query.to("cpu")))

        embeddings = query_embeddings[0].tolist()
        return embeddings

    # async def embed_images(self, images: list, batch_size: int = 4) -> AsyncGenerator[list[list[float]], None]:
    def embed_images(self, images: list[Image.Image], batch_size: int = 4) -> list[list[list[float]]]:
        """
        Embed a list of images using the model.
        """
        import torch
        # dataset = CustomImageDataset(images)

        # dataloader = DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     # collate_fn=lambda x: self.process_images(x),
        # )
        # print("dataloader:\n", dataloader)

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]
            batch_size_current = len(image_batch)

            print(i, type(image_batch), type(image_batch[0]))
            with torch.no_grad():
                # batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                batch_images = self.processor.process_images(image_batch).to(self.model.device)
                image_embeddings = self.model(**batch_images)

                # Shape: [batch_size, 1030, 128]
                # Convert embeddings_doc to CPU, then iterate over the batch dimension
                special_tokens = image_embeddings[:, 1024:, :]

                max_pool = torch.cat(
                    (
                        torch.max(
                            image_embeddings[:, :1024, :].reshape((batch_size_current, 32, 32, 128)),
                            dim=2,
                        ).values,
                        special_tokens,
                    ),
                    dim=1,
                )
                mean_pool = torch.cat(
                    (
                        torch.mean(
                            image_embeddings[:, :1024, :].reshape((batch_size_current, 32, 32, 128)),
                            dim=2,
                        ),
                        special_tokens,
                    ),
                    dim=1,
                )

                tensor1 = max_pool.cpu().float().numpy()
                tensor2 = mean_pool.cpu().float().numpy()
                tensor3 = image_embeddings.cpu().float().numpy()

                l1 = tensor1.tolist()
                l2 = tensor2.tolist()
                l3 = tensor3.tolist()

            # Yield each image's embeddings as a list of lists (1030 embeddings of 128 dimensions each)
            # for embedding_per_image in embeddings_list:
            #     yield embedding_per_image
            return l2  # returning mean_pool
