import base64
import json
import os
import re

from io import BytesIO
from typing import Any, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.models

from cachetools import cached, TTLCache
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from torchvision import transforms
from transformers import pipeline


AUTH_TOKEN = 'vEJh1cRQ4ZYKJK1udczu7PylW7Yp0KuEisUIQATzQtWbUCFZDOKTxmAyLfeA1uZt'

ASSETS_DIR = './assets'

MODEL_TTL = 300
MODEL_FACTORY_CACHE_SIZE = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def image_data_to_tensor(image_data, normalize: bool = True) -> torch.Tensor:
    image = Image.open(BytesIO(image_data))

    if normalize:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        transform = transforms.ToTensor()
    return transform(image)


def image_b64_to_tensor(image_b64: str, normalize: bool = True) -> torch.Tensor:
    image_data = base64.b64decode(re.sub(r'^data:image/\w+;base64,', '', image_b64))
    return image_data_to_tensor(image_data, normalize)


class ModelWrapper:
    def get_embedding(self, vectors: Any) -> Any:
        raise NotImplementedError

    def get_serialized_embedding(self, vectors: Any) -> str:
        return self.serialize_embedding(self.get_embedding(vectors))

    def serialize_embedding(self, embedding: Union[list[np.ndarray], list[torch.Tensor]]) -> str:
        if len(embedding) == 0:
            return '[]'
        if isinstance(embedding[0], np.ndarray) or isinstance(embedding[0], torch.Tensor):
            return json.dumps([e.tolist() for e in embedding])
        raise ValueError(f'Unsupported embedding type: {type(embedding)}')

    @classmethod
    def train(cls):
        raise NotImplementedError


class MobileNetWrapper(ModelWrapper):
    def __init__(self):
        self._weights = 'MobileNet_V3_Small_Weights.DEFAULT'
        mobilenet = torchvision.models.mobilenet_v3_small(weights=self._weights)
        self._mobilenet = torch.nn.Sequential(mobilenet.features, mobilenet.avgpool,
                                              torch.nn.Flatten())
        self._mobilenet.eval()
        self._mobilenet = self._mobilenet.to(device)

    def get_embedding(self, images: list[str]) -> list[torch.Tensor]:
        image_tensors = torch.cat([image_b64_to_tensor(image).unsqueeze(0) for image in images])
        return list(self._mobilenet(image_tensors.to(device)))

    @classmethod
    def train(cls):
        pass


class CountVectorizerWrapper(ModelWrapper):
    pretrain_path = f'{ASSETS_DIR}/count_vectorizer.joblib'

    def __init__(self):
        if not os.path.exists(type(self).pretrain_path):
            raise FileNotFoundError('CountVectorizer pretrain model not found')
        self._vectorizer: CountVectorizer = joblib.load(type(self).pretrain_path)

    def get_embedding(self, texts: Union[str, list[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._vectorizer.transform(texts).toarray()

    @classmethod
    def train(cls):
        vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        df = pd.read_pickle(f'{ASSETS_DIR}/movies.pkl.zst', usecols=['plot']).dropna()
        vectorizer.fit(df['overview'])
        joblib.dump(vectorizer, cls.pretrain_path)


class BertWrapper(ModelWrapper):
    def __init__(self):
        self._tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
        self._bert = pipeline(
            'feature-extraction', model='distilbert-base-uncased',
            tokenize_kwargs=self._tokenizer_kwargs,
            framework='pt', return_tensors=True, device=device
        )

    def _get_cls_embedding(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        return [e.squeeze(dim=0)[0, :] for e in embeddings]

    def get_embedding(self, texts: list[str]) -> list[torch.Tensor]:
        return self._get_cls_embedding(self._bert(texts))

    @classmethod
    def train(cls):
        pass


class ModelFactory:
    def __init__(self):
        self._models: dict[str, type[ModelWrapper]] = {}

    def add(self, name: str, wrapper_class: type[ModelWrapper]):
        self._models[name] = wrapper_class

    def remove(self, name: str):
        if name in self._models:
            del self._models[name]

    def get_class(self, name: str) -> type[ModelWrapper]:
        if name not in self._models:
            raise ValueError(f'Model {name} is unknown')
        return self._models[name]

    def get(self, name: str) -> ModelWrapper:
        return self.get_class(name)()


class MemoizedModelFactory(ModelFactory):
    @cached(cache=TTLCache(maxsize=MODEL_FACTORY_CACHE_SIZE, ttl=MODEL_TTL))
    def get(self, name: str) -> ModelWrapper:
        return super().get(name)


models = MemoizedModelFactory()
models.add('mobilenet', MobileNetWrapper)
models.add('count-vectorizer', CountVectorizerWrapper)
models.add('bert', BertWrapper)


class EmbeddingRequest(BaseModel):
    model: str
    data: list[str]


class TrainRequest(BaseModel):
    model: str
    token: str


app = FastAPI()


@app.post('/embedding/')
async def embedding(request: EmbeddingRequest):
    model = models.get(request.model)
    embedding = json.loads(model.get_serialized_embedding(request.data))
    return {'embedding': embedding}


@app.post('/train/')
async def train(request: TrainRequest):
    if request.token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail='Invalid authentication token')
    try:
        models.get_class(request.model).train()
        success = True
    except ValueError:
        success = False
    return {'success': success}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8057)
