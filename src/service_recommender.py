from typing import Optional

import aiohttp
import psycopg

from fastapi import FastAPI, HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel
from pymilvus import MilvusClient


MILVUS_URI = 'http://milvus:19530'
EMBEDDER_URI = 'http://embedder:8057'

pg_conn_kwargs = dict(
    dbname='movies',
    user='recommender',
    password='recommender',
    host='postgres',
    port=13006,
    row_factory=dict_row
)


class RecommendationRequest(BaseModel):
    data: str
    dtype: str
    model: str
    limit: Optional[int] = 5
    offset: Optional[int] = 0


app = FastAPI()


@app.post('/recommendation/')
async def recommendation(request: RecommendationRequest):
    milvus_client = MilvusClient(MILVUS_URI)
    pg_conn = await psycopg.AsyncConnection.connect(**pg_conn_kwargs)

    if request.dtype == 'title':
        # async with pg_conn.cursor() as cur:
        #     await cur.execute(
        #         'SELECT id FROM movies WHERE lower(title) = lower(%s) LIMIT 1',
        #         [request.data]
        #     )
        #     response = await cur.fetchall()
        # if not response:
        #     return {'recommendation': []}
        pass
    elif request.dtype in ['poster', 'plot']:
        embedder_query = {'model': request.model, 'data': [request.data]}
        async with aiohttp.ClientSession(EMBEDDER_URI) as embedder_client:
            async with embedder_client.post(f'/embedding', json=embedder_query) as response:
                embeddings = await response.json()
    else:
        raise HTTPException(status_code=400, detail='Invalid data type')
    
    with open('log.txt', 'a') as f:
        f.write('embeddings: ' + str(embeddings))

    similar_vectors = milvus_client.search(
        collection_name=f"embeddings_{request.model.replace('-', '_')}",
        data=[embeddings['embedding'][0]],
        limit=request.limit,
        offset=request.offset,
    )
    milvus_client.close()
    with open('log.txt', 'a') as f:
        f.write('similar_vectors: ' + str(similar_vectors))

    async with pg_conn.cursor() as cur:
        await cur.execute(
            """
            SELECT id, title, plot, 'data:image/jpg;base64,' || encode(poster, 'base64') as poster
            FROM movies
            WHERE id = ANY(%s)
            """,
            [[v['id'] for v in similar_vectors[0]]]
        )
        response = await cur.fetchall()
    await pg_conn.close()

    return {'recommendation ': response}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8058)
