import psycopg2
import pandas
import numpy
import sentence_transformers

df = pd.read_csv("/content/drive/MyDrive/Artifacts/vector_forge_artifacts/data.csv")

embeddings = np.load("/content/drive/MyDrive/Artifacts/vector_forge_artifacts/embeddings.npy")

df['embeddings'] = list(embeddings)

# Define your database connection parameters
host = "localhost"
user = "postgres"
password = "password"
port = 5432

# Establish a connection
conn = psycopg2.connect(
    host=host,
    user=user,
    password=password,
    port=port
)

cur = conn.cursor()

#install pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

#install pgvectorscale
cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;")
conn.commit()

# Register the vector type with psycopg2
register_vector(conn)

table_create_command = """
CREATE TABLE embeddings (
            id bigserial primary key, 
            title text,
            author text,
            content text,
            embedding vector(1536)
            );
            """

cur.execute(table_create_command)
cur.close()
conn.commit()

#Batch insert embeddings and metadata from dataframe into PostgreSQL database
register_vector(conn)
cur = conn.cursor()
# Prepare the list of tuples to insert
data_list = [(row['title'], row['author'], row['content'], np.array(row['embeddings'])) for index, row in df.iterrows()]
# Use execute_values to perform batch insertion
execute_values(cur, "INSERT INTO embeddings (title, url, content, tokens, embedding) VALUES %s", data_list)
# Commit after we insert all embeddings
conn.commit()

cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
num_records = cur.fetchone()[0]
print("Number of vector records in table: ", num_records,"\n")

cur.execute("SELECT * FROM embeddings LIMIT 1;")
records = cur.fetchall()
print("First record in table: ", records)

conn.close()