import argparse
from app.database import init_db, SessionLocal
from app.crud import import_data, create_indexes

def main():
    parser = argparse.ArgumentParser(description="Manage the vector search database")
    parser.add_argument("--initialize", action="store_true", help="Initialize the database tables and extensions")
    parser.add_argument("--import_data", action="store_true", help="Import data from CSV/JSON and embeddings")
    parser.add_argument("--create_indexes", action="store_true", help="Create database indexes")
    parser.add_argument("--data_path", type=str, help="Path to the data CSV/JSON file")
    parser.add_argument("--embeddings_path", type=str, help="Path to the embeddings numpy file")
    args = parser.parse_args()

    if args.initialize:
        print("Initializing database...")
        init_db()
        print("Database initialized.")

    if args.import_data:
        if not args.data_path or not args.embeddings_path:
            print("Error: --data_path and --embeddings_path are required for data import.")
        else:
            db = SessionLocal()
            try:
                import_data(db, args.data_path, args.embeddings_path)
            finally:
                db.close()

    if args.create_indexes:
        print("Creating database indexes...")
        create_indexes()

if __name__ == "__main__":
    main()
