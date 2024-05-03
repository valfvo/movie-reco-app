# movie-reco-app

To run the project, choose a directory to mount docker volumes, then compose up.

```
export DOCKER_VOLUMES_DIRECTORY="$HOME/movie-reco-app/volumes"
docker compose up
```

The initialization time should be less than 10 minutes. After that you can query the recommender service, as shown below.

```python
import requests
body = {
    'model': 'bert',
    'data': "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene..."
    'dtype': 'plot',
}
response = requests.post('http://localhost:8058/recommendation', json=body, timeout=20)
```

By default it will return the top 5 recommendations, but you can leverage Milvus to do pagination.
```python
body = {
    'model': 'mobilenet',
    'data': "data:image/jpg;base64,...",
    'dtype': 'poster',
    'limit': 20,
    'offset': 150,
}
```

Currently, the available models are `bert`, `count-vectorizer` and `mobilenet` with respective types `plot`, `plot`, `poster`. A `plot` must be a str of words while a `poster` must be a base64 encoded image.

## link to notebook on interpretability

https://colab.research.google.com/drive/1ccfvzIoipKaDKyxiUCExTIXf4-Z6MVys?usp=sharing
