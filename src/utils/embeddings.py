from typing import List
import torch


class SentenceEmbeddingEncoder:
    def __init__(self, model_name: str = "sentence-transformers/LaBSE", device: str = "cpu") -> None:
        # Import locally to avoid global import cost if unused
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        # Returns tensor of shape [B, D]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_tensor=True,
            device=str(self.device),
            normalize_embeddings=True,
        )
        # Clone to avoid returning an inference tensor which cannot be saved for backward
        return embeddings.clone().detach()


