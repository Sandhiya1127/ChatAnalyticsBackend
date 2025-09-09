# import pandas as pd
# from sqlalchemy import create_engine

# # Database connection setup
# db_config = {
#     'host': '139.59.12.79',
#     'database': 'updated_ui_db',
#     'user': 'appadmin',
#     'password': 'prowesstics',
#     'port': '5432'
# }
# connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
# engine = create_engine(connection_string)

# # Step 1: Load data from PostgreSQL
# query = 'SELECT * FROM public."overall_cleaned_base_and_pr_ef_policyef_with_reasons_bucket";'
# df = pd.read_sql(query, con=engine)

# # Ensure proper datetime formatting
# df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
# df['Policy End Date'] = pd.to_datetime(df['Policy End Date'], errors='coerce')
# df['Start Year'] = df['Policy Start Date'].dt.year
# df['Start Year-Month'] = df['Policy Start Date'].dt.to_period('M')
# df['End Year'] = df['Policy End Date'].dt.year
# df['End Year-Month'] = df['Policy End Date'].dt.to_period('M')

# df.dropna(subset=['CustomerID', 'Start Year', 'Policy No'], inplace=True)

# # Step 2: Remove duplicates for customer counts
# unique_customers = df.drop_duplicates(subset=['CustomerID', 'Start Year', 'Policy No'])

# # Step 3: Start Year-wise customer counts and policies
# start_year_metrics = (
#     unique_customers.groupby('Start Year')
#     .apply(lambda x: pd.Series({
#         'Total Customers': x['CustomerID'].nunique(),
#         'Total Policies': x['Policy No'].nunique(),
#         'New Customers': x[x['New Customers'] == 'Yes']['CustomerID'].nunique(),
#         'New Customers Total Policies': x[x['New Customers'] == 'Yes']['Policy No'].nunique(),
#         'Old Customers': x[x['New Customers'] == 'No']['CustomerID'].nunique(),
#         'Old Customers Total Policies': x[x['New Customers'] == 'No']['Policy No'].nunique()
#     }))
#     .reset_index()
# )

# # Step 4: Start Year-Month-wise customer counts and policies
# start_month_metrics = (
#     unique_customers.groupby('Start Year-Month')
#     .apply(lambda x: pd.Series({
#         'Total Customers': x['CustomerID'].nunique(),
#         'Total Policies': x['Policy No'].nunique(),
#         'New Customers': x[x['New Customers'] == 'Yes']['CustomerID'].nunique(),
#         'New Customers Total Policies': x[x['New Customers'] == 'Yes']['Policy No'].nunique(),
#         'Old Customers': x[x['New Customers'] == 'No']['CustomerID'].nunique(),
#         'Old Customers Total Policies': x[x['New Customers'] == 'No']['Policy No'].nunique()
#     }))
#     .reset_index()
# )

# # Step 5: Open Policies metrics by End Year-Month
# open_policies_metrics = (
#     df[df['Policy Status'] == 'Open']
#     .groupby('End Year-Month')['CustomerID']
#     .nunique()
#     .reset_index(name='Open Policies Count (Year-Month)')
# )

# # Step 6: Renewed and Not Renewed Policies Metrics by End Year
# renewal_year_metrics = (
#     df.groupby('End Year')
#     .apply(lambda x: pd.Series({
#         'Renewed Policies Count': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique(),
#         'Not Renewed Policies Count': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique(),
#         'Renewed Policies Avg': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#         'Not Renewed Policies Avg': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#     }))
#     .reset_index()
# )

# # Step 7: Renewed and Not Renewed Policies Metrics by End Year-Month
# renewal_month_metrics = (
#     df.groupby('End Year-Month')
#     .apply(lambda x: pd.Series({
#         'Renewed Policies Count': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique(),
#         'Not Renewed Policies Count': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique(),
#         'Renewed Policies Avg': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#         'Not Renewed Policies Avg': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#     }))
#     .reset_index()
# )

# # Step 8: State-wise, End-Year-wise Renewed/Not Renewed Metrics
# state_end_year_metrics = (
#     df.groupby(['Cleaned_state2', 'End Year'])
#     .apply(lambda x: pd.Series({
#         'Renewed Policies Count': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique(),
#         'Not Renewed Policies Count': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique(),
#         'Renewed Policies Avg': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#         'Not Renewed Policies Avg': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#     }))
#     .reset_index()
# )

# state_end_month_metrics = (
#     df.groupby(['Cleaned_state2', 'End Year-Month'])
#     .apply(lambda x: pd.Series({
#         'Renewed Policies Count': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique(),
#         'Not Renewed Policies Count': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique(),
#         'Renewed Policies Avg': x[x['Policy Status'] == 'Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#         'Not Renewed Policies Avg': x[x['Policy Status'] == 'Not Renewed']['Policy No'].nunique() / x['Policy No'].nunique() if x['Policy No'].nunique() > 0 else 0,
#     }))
#     .reset_index()
# )

# # Deduplicate data for total customers and churn analysis by End Year
# churn_data_year = df[df['Churn Label'] == 'Yes'].drop_duplicates(subset=['CustomerID', 'End Year'])
# deduplicated_data_year = df.drop_duplicates(subset=['CustomerID', 'End Year'])

# # Step 9: End-Year-wise Churned Customer Metrics with Total Customers
# churn_year_metrics = (
#     deduplicated_data_year.groupby('End Year')
#     .apply(lambda x: pd.Series({
#         'Total Customers': x['CustomerID'].nunique(),
#         'Churned Customers Count': churn_data_year[churn_data_year['End Year'] == x.name]['CustomerID'].nunique(),
#         'Churned Customers Avg': churn_data_year[churn_data_year['End Year'] == x.name]['CustomerID'].nunique() / x['CustomerID'].nunique() if x['CustomerID'].nunique() > 0 else 0
#     }))
#     .reset_index()
# )

# # Deduplicate data for total customers and churn analysis by End Year-Month
# churn_data_month = df[df['Churn Label'] == 'Yes'].drop_duplicates(subset=['CustomerID', 'End Year-Month'])
# deduplicated_data_month = df.drop_duplicates(subset=['CustomerID', 'End Year-Month'])

# # Step 10: End-Year-Month-wise Churned Customer Metrics with Total Customers
# churn_month_metrics = (
#     deduplicated_data_month.groupby('End Year-Month')
#     .apply(lambda x: pd.Series({
#         'Total Customers': x['CustomerID'].nunique(),
#         'Churned Customers Count': churn_data_month[churn_data_month['End Year-Month'] == x.name]['CustomerID'].nunique(),
#         'Churned Customers Avg': churn_data_month[churn_data_month['End Year-Month'] == x.name]['CustomerID'].nunique() / x['CustomerID'].nunique() if x['CustomerID'].nunique() > 0 else 0
#     }))
#     .reset_index()
# )

# # Save all metrics to Excel
# file_path = 'EDA_Metrics(base&Pr).xlsx'
# with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
#     start_year_metrics.to_excel(writer, index=False, sheet_name='Start Year Metrics')
#     start_month_metrics.to_excel(writer, index=False, sheet_name='Start Month Metrics')
#     open_policies_metrics.to_excel(writer, index=False, sheet_name='Open Policies Metrics')
#     renewal_year_metrics.to_excel(writer, index=False, sheet_name='Renewal Year Metrics')
#     renewal_month_metrics.to_excel(writer, index=False, sheet_name='Renewal Month Metrics')
#     state_end_year_metrics.to_excel(writer, index=False, sheet_name='State-End Year Metrics')
#     state_end_month_metrics.to_excel(writer, index=False, sheet_name='State-End Month Metrics')
#     churn_year_metrics.to_excel(writer, index=False, sheet_name='Churn Year Metrics')
#     churn_month_metrics.to_excel(writer, index=False, sheet_name='Churn Month Metrics')

# print(f"Metrics saved to {file_path}")


import pandas as pd
from sqlalchemy import create_engine
import sweetviz

engine = create_engine("postgresql://appadmin:prowesstics@139.59.12.79:5432/updated_ui_db")

query = 'SELECT * FROM '
df = pd.read_sql(query, engine)

report = sweetviz.analyze(df)
report.show_html("EDA_Report.html")

# sudharshan
# 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