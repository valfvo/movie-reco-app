import json
import time

from dataclasses import dataclass

import joblib
import numpy as np

from io import BytesIO
from minio import Minio
from pymilvus import BulkInsertState, connections, DataType, MilvusClient, utility
from safetensors.torch import load_file


ASSETS_DIR = './assets'
MILVUS_URI = 'http://milvus:19530'

plot_ids = np.load(f'{ASSETS_DIR}/plot_ids.npy')
poster_ids = np.load(f'{ASSETS_DIR}/poster_ids.npy')

milvus_bucket = 'a-bucket'
milvus_root_path = 'files'

connections.connect(host='milvus', port='19530')
milvus_client = MilvusClient(MILVUS_URI)
minio_client = Minio('minio:9000', access_key='minioadmin', secret_key='minioadmin',
                     secure=False)


@dataclass
class EmbeddingCollection:
    model_name: str
    file: str
    ids: np.ndarray

    @property
    def name(self):
        return f"embeddings_{self.model_name.replace('-', '_')}"


collections = [
    EmbeddingCollection('mobilenet', 'embeddings_mobilenet.safetensors', poster_ids),
    EmbeddingCollection('count-vectorizer', 'embeddings_count_vectorizer.joblib.gz', plot_ids),
    EmbeddingCollection('bert', 'embeddings_cls_bert.safetensors', plot_ids)
]

uploaded_collections = milvus_client.list_collections()
collections = [c for c in collections if c.name not in uploaded_collections]

for c in collections:
    if c.file.endswith('.joblib.gz'):
        embeddings = joblib.load(f'{ASSETS_DIR}/{c.file}')
    elif c.file.endswith('.safetensors'):
        embeddings = load_file(f'{ASSETS_DIR}/{c.file}')['embedding']
    else:
        raise ValueError(f'Invalid file type: {c.file}')
    embedding_dim = embeddings.shape[1]

    embedding_str = json.dumps({
        'rows': [{'id': int(i), 'vector': e.tolist()} for i, e in zip(c.ids, embeddings)]
    })
    embedding_bytes = BytesIO(embedding_str.encode('utf-8'))

    # minio_client.bucket_exists(milvus_bucket)
    minio_client.put_object(
        milvus_bucket, f"{milvus_root_path}/{c.name}.json",
        embedding_bytes, len(embedding_str)
    )

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name='vector', datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(field_name='id', index_type='STL_SORT')
    index_params.add_index(field_name='vector', index_type='IVF_FLAT',
                           metric_type='COSINE', params={'nlist': 128})

    # milvus_client.create_collection(collection_name='embeddings_', dimension=embedding_dim)
    milvus_client.create_collection(collection_name=c.name,
                                    schema=schema, index_params=index_params)

    task_id = utility.do_bulk_insert(collection_name=c.name,
                                     files=[f'{milvus_root_path}/{c.name}.json'])

    i = 0
    max_checks = 30
    state = BulkInsertState.ImportPending
    pending_states = [BulkInsertState.ImportPending, BulkInsertState.ImportStarted]

    while i < max_checks and state in pending_states:
        state = utility.get_bulk_insert_state(task_id).state
        time.sleep(1)
        i += 1

milvus_client.close()
