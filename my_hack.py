import os
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
import torch


# add callback input argument to log eval plot
class MySentenceTransformersFinetuneEngine(SentenceTransformersFinetuneEngine):
    def __init__(self, loss=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if loss is not None:
            self.loss = loss(self.model)
        else:
            self.loss = MultipleNegativesRankingLoss(self.model)

    def finetune(self, callback, **train_kwargs):
        super().finetune(callback=callback, **train_kwargs)


class MyEmbeddingAdapterFinetuneEngine(EmbeddingAdapterFinetuneEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def finetune(self, callback, **train_kwargs):
        super().finetune(callback=callback, **train_kwargs)


class MySentenceTransformerWrapper(SentenceTransformer):
    def __init__(self, adapter_embedding_model):
        super().__init__()
        self.adapter_embedding_model = adapter_embedding_model

    def encode(self, sentences, batch_size=32, convert_to_tensor=False, *args, **kwargs):
        embeddings = []
        for sentence in sentences:
            embedding = self.adapter_embedding_model._get_query_embedding(sentence)
            embeddings.append(embedding)
        if convert_to_tensor:
            embeddings = torch.tensor(embeddings).to(self.device)
        return embeddings


