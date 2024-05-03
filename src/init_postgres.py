import pandas as pd
import psycopg

from psycopg.rows import dict_row
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql


ASSETS_DIR = './assets'
pg_conn_kwargs = dict(
    dbname='movies',
    user='recommender',
    password='recommender',
    host='postgres',
    port=13006,
    row_factory=dict_row
)

def init_postgres():
    pg_conn = psycopg.Connection.connect(**pg_conn_kwargs)

    table_exists = False
    with pg_conn.cursor() as cur:
        try:
            cur.execute('select count(*) from movies')
            table_exists = True
        except psycopg.errors.UndefinedTable:
            pass  # OK, table does not exist
    pg_conn.close()

    if table_exists:
        return
    else:
        pg_conn = psycopg.Connection.connect(**pg_conn_kwargs)

    df = pd.read_pickle(f'{ASSETS_DIR}/movies.pkl.zst')

    with pg_conn.cursor() as cur:
        cur.execute(
            """CREATE TABLE movies (
                id            integer CONSTRAINT movie_id PRIMARY KEY,
                title         text NOT NULL,
                release_date  date,
                genres        text[],
                plot          text,
                poster        bytea
            );"""
        )
    pg_conn.commit()

    pg_engine = create_engine(
        'postgresql+psycopg://recommender:recommender@postgres:13006/movies'
    )

    table_dtype = {
        'id':            postgresql.INTEGER,
        'title':         postgresql.TEXT,
        'release_date':  postgresql.DATE,
        'genres':        postgresql.ARRAY(postgresql.TEXT),
        'plot':          postgresql.TEXT,
        'poster':        postgresql.BYTEA
    }

    df.to_sql('movies', pg_engine, index=True, index_label='id', dtype=table_dtype,
              method='multi', chunksize=5000, if_exists='replace')
    pg_conn.close()


if __name__ == '__main__':
    init_postgres()
