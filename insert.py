from psycopg2.extras import execute_values
import numpy as np
import time

def insert_embeddings(conn, df, batch_size=10000):
    try:
        total_rows = len(df)
        print(f"Inserting {total_rows} embeddings in batches of {batch_size}...")
        start_time = time.time()
        inserted = 0

        with conn.cursor() as cur:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                data_list = [
                    (row['title'], row['author'], row['description'], np.array(row['embeddings']).tolist())
                    for _, row in batch.iterrows()
                ]

                insert_query = """
                INSERT INTO embeddings (title, author, content, embedding) 
                VALUES %s
                """
                try:
                    execute_values(cur, insert_query, data_list)
                    conn.commit()
                    inserted += len(data_list)
                    print(f"✅ Inserted batch {i // batch_size + 1}: {len(data_list)} rows (Total: {inserted})")
                except Exception as e:
                    print(f"⚠️ Error inserting batch {i // batch_size + 1}: {e}")
                    conn.rollback()

        print(f"Finished inserting in {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Fatal error during insertion: {e}")
        conn.rollback()


def vacuum_embeddings(conn):
    # Save current autocommit state
    old_autocommit = conn.autocommit
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("VACUUM embeddings;")
            print("✅ VACUUM completed: IVF index is now up-to-date.")
    finally:
        conn.autocommit = old_autocommit  # Restore previous state