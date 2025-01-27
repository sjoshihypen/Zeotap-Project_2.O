import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Define the file paths (using raw strings to avoid escape sequence warnings)
customers_file_path = r'D:\Zeotap\Datasets\Customers.csv'
products_file_path = r'D:\Zeotap\Datasets\Products.csv'
transactions_file_path = r'D:\Zeotap\Datasets\Transactions.csv'

# Check if the files exist
if not os.path.exists(customers_file_path):
    raise FileNotFoundError(f"File not found: {customers_file_path}")
if not os.path.exists(products_file_path):
    raise FileNotFoundError(f"File not found: {products_file_path}")
if not os.path.exists(transactions_file_path):
    raise FileNotFoundError(f"File not found: {transactions_file_path}")

# Load the data (with encoding to avoid UnicodeDecodeError and skip bad lines)
customers_df = pd.read_csv(customers_file_path, encoding='ISO-8859-1')
products_df = pd.read_csv(products_file_path, encoding='ISO-8859-1')
transactions_df = pd.read_csv(transactions_file_path, encoding='ISO-8859-1', on_bad_lines='skip', engine='python')

# Clean column names to remove unwanted characters or BOM markers
customers_df.columns = customers_df.columns.str.strip().str.replace('ï»¿', '')
products_df.columns = products_df.columns.str.strip().str.replace('ï»¿', '')
transactions_df.columns = transactions_df.columns.str.strip().str.replace('ï»¿', '')

# Debug: Print column names to ensure they match
print("Customers DataFrame Columns:", customers_df.columns)
print("Products DataFrame Columns:", products_df.columns)
print("Transactions DataFrame Columns:", transactions_df.columns)

# Verify 'CustomerID' exists in both DataFrames
if 'CustomerID' not in customers_df.columns:
    raise KeyError("CustomerID column missing from Customers DataFrame")
if 'CustomerID' not in transactions_df.columns:
    raise KeyError("CustomerID column missing from Transactions DataFrame")

# Verify 'ProductID' exists in both DataFrames
if 'ProductID' not in transactions_df.columns:
    raise KeyError("ProductID column missing from Transactions DataFrame")
if 'ProductID' not in products_df.columns:
    raise KeyError("ProductID column missing from Products DataFrame")

# Merge customer and transaction data
transaction_data = pd.merge(transactions_df, customers_df, on='CustomerID')

# Debug: Print the merged data preview
print("Merged Transaction Data Preview:")
print(transaction_data.head())

# Debug: Print the products data preview
print("Products Data Preview:")
print(products_df.head())

# Feature engineering: Aggregate transaction data for each customer
customer_transactions = transaction_data.groupby('CustomerID').agg(
    total_spent=pd.NamedAgg(column='TotalValue', aggfunc='sum'),
    num_transactions=pd.NamedAgg(column='TransactionID', aggfunc='count')
).reset_index()

# Merge with product data to get product-related features
product_data = pd.merge(transaction_data, products_df, on='ProductID')

# Debug: Print the merged product data preview
print("Merged Product Data Preview:")
print(product_data.head())

# Create additional features like total quantity purchased for each category
category_purchase = product_data.groupby(['CustomerID', 'Category']).agg(
    total_quantity=pd.NamedAgg(column='Quantity', aggfunc='sum')
).reset_index()

# Pivot to create a customer-item matrix
category_matrix = category_purchase.pivot_table(
    index='CustomerID', columns='Category', values='total_quantity', fill_value=0
)

# Combine customer features (demographic and transaction-related)
customer_features = pd.merge(customer_transactions, category_matrix, on='CustomerID', how='left')

# Handle missing values in customer_features (if any)
customer_features = customer_features.fillna(0)

# Normalize the feature data
scaler = StandardScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(customer_features.drop('CustomerID', axis=1)), 
    columns=customer_features.columns[1:]
)

# Compute similarity between customers using cosine similarity
cosine_sim = cosine_similarity(normalized_features)

# Debug: Print the shape of the similarity matrix
print("Shape of Cosine Similarity Matrix:", cosine_sim.shape)

# Create a function to get top 3 lookalikes for each customer
def get_top_lookalikes(cosine_sim, customer_ids, top_n=3):
    lookalikes = {}
    for i, customer_id in enumerate(customer_ids):
        # Get the indices of the top N similar customers (excluding the customer itself)
        similar_indices = cosine_sim[i].argsort()[-(top_n + 1):-1][::-1]
        
        # Filter out indices that are out of bounds for the customer_ids list
        similar_indices = [idx for idx in similar_indices if idx < len(customer_ids)]
        
        similar_customers = [(customer_ids[idx], cosine_sim[i][idx]) for idx in similar_indices]
        lookalikes[customer_id] = similar_customers
    return lookalikes

# Get the customer IDs for the first 20 customers
top_20_customers = customer_features['CustomerID'].head(20).tolist()

# Debug: Print the number of customers
print("Number of Customers:", len(top_20_customers))

# Get top 3 lookalikes for each of the first 20 customers
lookalikes = get_top_lookalikes(cosine_sim, top_20_customers)

# Create a DataFrame for the output
lookalike_df = []
for cust_id, recommendations in lookalikes.items():
    for rec in recommendations:
        lookalike_df.append([cust_id, rec[0], rec[1]])

lookalike_df = pd.DataFrame(lookalike_df, columns=['CustomerID', 'LookalikeCustomerID', 'SimilarityScore'])

# Save to CSV file
lookalike_df.to_csv('Lookalike.csv', index=False)

# Display top 3 lookalikes for the first 20 customers
print(lookalike_df)