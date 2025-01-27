import pandas as pd

# Load the transactions dataset
transactions = pd.read_csv(r'D:\Zeotap\Datasets\Transactions.csv')

# Load the customer dataset
customers = pd.read_csv(r'D:\Zeotap\Datasets\Customers.csv')

# Fix the encoding issue with 'CustomerID' in the transactions dataset (remove the BOM prefix)
transactions.rename(columns={'ï»¿CustomerID': 'CustomerID'}, inplace=True)

# Print the available columns in both datasets for verification
print("Available columns in Transactions dataset:", transactions.columns.tolist())
print("Available columns in Customers dataset:", customers.columns.tolist())

# Merge the transactions and customers datasets on 'CustomerID'
merged_data = pd.merge(transactions, customers, on='CustomerID', how='left')

# Check if required columns exist in the merged dataset
required_columns = ['CustomerID', 'CustomerName', 'Region', 'SignupDate']
missing_columns = [col for col in required_columns if col not in merged_data.columns]

if missing_columns:
    print(f"The following columns are missing after merging: {missing_columns}")
    print("Please check your datasets and ensure the columns exist.")
else:
    # Perform the aggregation on the merged dataset
    transactions_summary = merged_data.groupby('CustomerID').agg({
        'CustomerName': 'first',  # Get the first occurrence of CustomerName
        'Region': 'first',        # Get the first occurrence of Region
        'SignupDate': 'first',    # Get the first occurrence of SignupDate
        'TransactionID': 'count', # Count the number of transactions per customer
        'TotalValue': 'sum',      # Sum the total value of transactions per customer
        # Add more columns and aggregation functions as needed
    })

    # Display the summary
    print(transactions_summary)