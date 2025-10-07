import re
import difflib
from sqlalchemy import create_engine, text

# Connect to your PostgreSQL database
from sqlalchemy import create_engine
import os

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")  # must be int
DB_NAME = os.getenv("POSTGRES_DB")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def get_columns(schema, table):
    """Fetch all column names for a table."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = :table
        """), {"schema": schema, "table": table})
        return [row[0] for row in result]

def suggest_column(unknown_col, table_cols):
    """Return the best match for a column name."""
    matches = difflib.get_close_matches(unknown_col, table_cols, n=1, cutoff=0.6)
    return matches[0] if matches else None

def find_table_aliases(query):
    """
    Returns mapping of alias -> (schema, table)
    Example: 'FROM stage.loan_main_2024 lm' -> {'lm': ('stage', 'loan_main_2024')}
    """
    pattern = re.compile(r'FROM\s+([\w\.]+)(?:\s+AS)?\s+(\w+)?', re.IGNORECASE)
    joins_pattern = re.compile(r'JOIN\s+([\w\.]+)(?:\s+AS)?\s+(\w+)?', re.IGNORECASE)
    tables = {}
    for m in pattern.findall(query) + joins_pattern.findall(query):
        full_table = m[0]
        alias = m[1] or full_table.split('.')[-1]
        if '.' in full_table:
            schema, table = full_table.split('.')
        else:
            schema, table = 'public', full_table
        tables[alias] = (schema, table)
    return tables

def auto_fix_sql_multiple_tables(query):
    """Automatically replace undefined columns in multi-table SQL."""
    table_aliases = find_table_aliases(query)
    
    # Collect all columns for all tables
    all_columns = {}
    for alias, (schema, table) in table_aliases.items():
        all_columns[alias] = get_columns(schema, table)
    
    # Find all potential column references in query (alias.column or just column)
    col_refs = re.findall(r'\b([\w_]+)\.([\w_]+)\b|\b([\w_]+)\b', query)
    
    replacements = {}
    for full in col_refs:
        if full[0] and full[1]:  # alias.column
            alias, col = full[0], full[1]
            if alias in all_columns and col not in all_columns[alias]:
                suggestion = suggest_column(col, all_columns[alias])
                if suggestion:
                    replacements[f"{alias}.{col}"] = f"{alias}.{suggestion}"
        elif full[2]:  # column without alias
            col = full[2]
            # Try to match in any table
            found = False
            for alias, cols in all_columns.items():
                if col in cols:
                    found = True
                    break
            if not found:
                # Suggest from any table
                suggestion = None
                for alias, cols in all_columns.items():
                    suggestion = suggest_column(col, cols)
                    if suggestion:
                        replacements[col] = suggestion
                        break
    
    # Replace all typos in query
    fixed_query = query
    for wrong, correct in replacements.items():
        fixed_query = re.sub(rf'\b{re.escape(wrong)}\b', correct, fixed_query)
    
    return fixed_query, replacements

# Example SQL with typos across tables
sql = """
SELECT lm.fslopener_id, lm.term, lm.chargeoffs, lm.amountfinanced,
       app.fsl_preselectiondate
FROM stage.loan_main_2024 lm
JOIN stage.app_main_2024 app ON lm.merchantid = app.merchantid
WHERE lm.fslopener_id IS NOT NULL
"""

fixed_sql, corrections = auto_fix_sql_multiple_tables(sql)

print("Suggested corrections:", corrections)
print("Fixed SQL:\n", fixed_sql)
