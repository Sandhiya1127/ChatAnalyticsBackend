from django.db import connection

# ---------------------------
# Helper functions
# ---------------------------
def get_table_columns(schema_name, table_name):
    """Return all column names for a given schema and table."""
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
        """, [schema_name, table_name])
        return [row[0] for row in cursor.fetchall()]

def get_safe_column(schema_name, table_name, preferred_columns):
    """Return the first existing column from preferred_columns list."""
    existing_columns = get_table_columns(schema_name, table_name)
    for col in preferred_columns:
        if col in existing_columns:
            return col
    raise ValueError(f"None of the preferred columns exist in {schema_name}.{table_name}")

def build_numeric_clean_case(column_name):
    """
    Returns a SQL CASE statement to safely clean numeric values.
    Handles NULL, empty, NaN, N/A, NULL strings, etc.
    """
    return f"""
        CASE
            WHEN {column_name} IS NULL THEN NULL
            WHEN TRIM({column_name}::text) = '' THEN NULL
            WHEN UPPER(TRIM({column_name}::text)) IN ('NAN', 'NULL', 'N/A', 'NA') THEN NULL
            WHEN TRIM({column_name}::text) ~ '^-?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?$' 
                THEN {column_name}::NUMERIC
            ELSE NULL
        END
    """

# ---------------------------
# Main dynamic SQL generator
# ---------------------------
def generate_schema_agnostic_sql(
    schema_name='stage',
    app_table='app_main_2024',
    loan_table='loan_main_2024',
    group_column_candidates=None,
    numeric_column_candidates=None,
    aggregation_function='SUM',  # SUM, AVG, MAX, MIN
    filters=None,  # dict of {column_name: value} for WHERE clause
    year_column='created_at',  # optional column for year filter
    year_value=None
):
    """
    Generate fully dynamic SQL with:
    - Dynamic grouping column
    - Dynamic numeric column
    - Any aggregation function
    - Optional filters
    - Automatic numeric cleaning
    """

    if group_column_candidates is None:
        group_column_candidates = ['vantagescore_bucket', 'vantage_score_bucket', 'vantage_bucket']
    if numeric_column_candidates is None:
        numeric_column_candidates = ['loan_interest', 'loanamount', 'interest_rate']

    # Detect columns dynamically
    safe_group_column = get_safe_column(schema_name, app_table, group_column_candidates)
    safe_numeric_column = get_safe_column(schema_name, loan_table, numeric_column_candidates)

    # Build WHERE conditions
    where_clauses = ["TRUE"]  # Start with TRUE for easy AND chaining
    params = []

    if filters:
        for col, val in filters.items():
            where_clauses.append(f'a.{col} = %s')
            params.append(val)

    if year_value and year_column:
        where_clauses.append(f"EXTRACT(YEAR FROM CAST(a.{year_column} AS DATE)) = %s")
        params.append(year_value)

    where_sql = " AND ".join(where_clauses)

    # Build aggregation CASE safely
    clean_numeric_case = build_numeric_clean_case(f'l.{safe_numeric_column}')

    # Construct full dynamic SQL
    sql = f"""
        SELECT
            a.{safe_group_column} AS group_value,
            {aggregation_function}({clean_numeric_case}) AS aggregated_value
        FROM "{schema_name}"."{app_table}" AS a
        INNER JOIN "{schema_name}"."{loan_table}" AS l
            ON a.customerid = l.customerid
        WHERE {where_sql}
        GROUP BY a.{safe_group_column}
        ORDER BY a.{safe_group_column};
    """

    return sql.strip(), params

# ---------------------------
# Execution function
# ---------------------------
def execute_dynamic_sql(sql, params=None):
    """Execute dynamic SQL and return results as list of dicts."""
    with connection.cursor() as cursor:
        cursor.execute(sql, params or [])
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example: Aggregate SUM of loan_interest grouped by vantagescore_bucket for merchant 'Petland' in 2024
    filters = {"merchantid": "Petland"}  # can add more filters dynamically
    sql, params = generate_schema_agnostic_sql(
        aggregation_function='SUM',
        filters=filters,
        year_value=2024
    )
    results = execute_dynamic_sql(sql, params)
    for row in results:
        print(row)
