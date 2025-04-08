# VectorSeek

## Setup a postgresDB using docker
- `docker run --name postgres-container -p 5432:5432 -e POSTGRES_PASSWORD=yourPassword -d ankane/pgvector`
- `docker exec -it postgres-container bash`
- `psql -h localhost -p 5432 -U postgres`
