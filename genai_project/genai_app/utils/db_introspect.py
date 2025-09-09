
from sqlalchemy import inspect

def extract_schema_from_sqlalchemy(engine):
    inspector = inspect(engine)
    blocks = []

    for schema in inspector.get_schema_names():
        for table in inspector.get_table_names(schema=schema):
            cols = inspector.get_columns(table, schema=schema)
            col_text = "\n".join([f"- {c['name']}: {c['type']}" for c in cols])
            blocks.append(f"Schema: {schema}\nTable: {table}\n{col_text}")

    return "\n\n".join(blocks)
