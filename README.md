# VectorSeek

## Setup a postgresDB using docker
- `docker run --name postgres-container -p 5432:5432 -e POSTGRES_PASSWORD=password -d ankane/pgvector`

- `docker run --name postgres-container \
  --shm-size=1g \
  --memory=4g \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  -v pgdata:/var/lib/postgresql/data \
  -d ankane/pgvector`

### Accessing postgres cli
- `docker exec -it postgres-container bash`
- `psql -h localhost -p 5432 -U postgres`

## Fetching data
- use gdown to fetch data from gdrive
- `gdown --folder https://drive.google.com/drive/folders/1-Ut1psQWdl8ZVlFAKHs-vuoysGQ8n0vU`
- update the `DATA_PATH` and `EMBEDDINGS_PATH` in `config.py`

## Running the app
- use `app.py` run the fastAPI server
- use `main.py` to test the functions
    - you'll need to initialise the table and insert data on first run
    - after that you can directly run the `perform_vector_search` function