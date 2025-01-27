import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
import csv

# Load the datasets with proper file paths and error handling
try:
    # Load Customers CSV
    customers = pd.read_csv(
        r'D:\Zeotap\Datasets\Customers.csv',
        encoding='latin1',
        engine='python',
        quotechar='"',
        skipinitialspace=True
    )
    
    # Load Products CSV
    products = pd.read_csv(
        r'D:\Zeotap\Datasets\Products.csv',
        encoding='latin1',
        engine='python',
        quotechar='"',
        skipinitialspace=True
    )
    
    # Load Transactions CSV with error handling for malformed lines
    transactions = pd.read_csv(
        r'D:\Zeotap\Datasets\Transactions.csv',
        encoding='latin1',
        engine='python',
        quotechar='"',
        skipinitialspace=True,
        quoting=csv.QUOTE_MINIMAL,  # Ensures correct handling of quotes
        on_bad_lines='skip'  # Skip any problematic lines
    )

except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")
    exit()  # Stop execution if CSV reading fails

# Clean column names by stripping leading/trailing spaces
customers.columns = customers.columns.str.strip()
products.columns = products.columns.str.strip()
transactions.columns = transactions.columns.str.strip()

# Data Preprocessing
# Convert columns to appropriate datatypes
try:
    customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
except KeyError:
    print("Error: 'SignupDate' column not found in the Customers dataset.")
    exit()

try:
    # Check if 'TransactionDate' column exists in the transactions dataset
    if 'TransactionDate' not in transactions.columns:
        # Try to find a similar column name (e.g., with leading/trailing spaces or different case)
        similar_columns = [col for col in transactions.columns if 'transactiondate' in col.lower()]
        if similar_columns:
            # Use the first similar column found
            transactions['TransactionDate'] = pd.to_datetime(transactions[similar_columns[0]])
        else:
            raise KeyError("'TransactionDate' column not found in the transactions dataset.")
    else:
        transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
except KeyError as e:
    print(f"Error: {e}")
    exit()

# Check if 'CustomerID' exists in both datasets
if 'CustomerID' not in transactions.columns or 'CustomerID' not in customers.columns:
    # Try to find similar column names (case-insensitive)
    transactions_customer_col = [col for col in transactions.columns if 'customerid' in col.lower()]
    customers_customer_col = [col for col in customers.columns if 'customerid' in col.lower()]
    
    if transactions_customer_col and customers_customer_col:
        # Use the first similar column found
        transactions.rename(columns={transactions_customer_col[0]: 'CustomerID'}, inplace=True)
        customers.rename(columns={customers_customer_col[0]: 'CustomerID'}, inplace=True)
    else:
        raise KeyError("'CustomerID' column not found in one or both datasets.")
else:
    # Ensure 'CustomerID' column names are consistent
    transactions.rename(columns={'CustomerID': 'CustomerID'}, inplace=True)
    customers.rename(columns={'CustomerID': 'CustomerID'}, inplace=True)

# Check if 'ProductID' exists in both datasets
if 'ProductID' not in transactions.columns or 'ProductID' not in products.columns:
    # Try to find similar column names (case-insensitive)
    transactions_product_col = [col for col in transactions.columns if 'productid' in col.lower()]
    products_product_col = [col for col in products.columns if 'productid' in col.lower()]
    
    if transactions_product_col and products_product_col:
        # Use the first similar column found
        transactions.rename(columns={transactions_product_col[0]: 'ProductID'}, inplace=True)
        products.rename(columns={products_product_col[0]: 'ProductID'}, inplace=True)
    else:
        raise KeyError("'ProductID' column not found in one or both datasets.")
else:
    # Ensure 'ProductID' column names are consistent
    transactions.rename(columns={'ProductID': 'ProductID'}, inplace=True)
    products.rename(columns={'ProductID': 'ProductID'}, inplace=True)

# Merge the data to get a unified dataset
try:
    merged_data = pd.merge(transactions, customers, on='CustomerID', how='inner')
    merged_data = pd.merge(merged_data, products, on='ProductID', how='inner')
except KeyError as e:
    print(f"Error during merging datasets: {e}")
    exit()

# Exploratory Data Analysis
# 1. General Info and Missing Values
print("General Info of Merged Data:")
print(merged_data.info())

print("\nMissing Values:")
print(merged_data.isnull().sum())

# 2. Descriptive Statistics
print("\nDescriptive Statistics:")
print(merged_data.describe())

# 3. Business Insights

# Insight 1: Total Revenue per Region
region_revenue = merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
print("\nTotal Revenue per Region:")
print(region_revenue)

# Insight 2: Top 10 Popular Products
top_products = merged_data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Popular Products by Quantity Sold:")
print(top_products)

# Insight 3: Monthly Sales Trend
monthly_sales = merged_data.groupby(merged_data['TransactionDate'].dt.to_period('M'))['TotalValue'].sum()
print("\nMonthly Sales Trend:")
print(monthly_sales)

# Insight 4: Average Order Value by Customer
avg_order_value = merged_data.groupby('CustomerID')['TotalValue'].sum().mean()
print("\nAverage Order Value by Customer:")
print(avg_order_value)

# Insight 5: Product Category Revenue Contribution
category_revenue = merged_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
print("\nRevenue by Product Category:")
print(category_revenue)

# Plotting for visualization
# Total Revenue per Region
plt.figure(figsize=(10, 6))
sns.barplot(x=region_revenue.index, y=region_revenue.values, palette='Blues_d')
plt.title('Total Revenue by Region')
plt.ylabel('Total Revenue (USD)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top 10 Popular Products by Quantity
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title('Top 10 Popular Products by Quantity Sold')
plt.ylabel('Quantity Sold')
plt.tight_layout()
plt.show()

# Monthly Sales Trend
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', color='green', marker='o')
plt.title('Monthly Sales Trend')
plt.ylabel('Total Sales Value (USD)')
plt.tight_layout()
plt.show()

# PDF Report Generation
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Business Insights Report', 0, 1, 'C')
        self.ln(10)
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

# Create PDF report
pdf = PDF()
pdf.add_page()

# Add Insights to PDF
pdf.chapter_title('Business Insight 1: Total Revenue per Region')
pdf.chapter_body(f"Total Revenue per Region:\n{region_revenue.to_string()}\n")

pdf.chapter_title('Business Insight 2: Top 10 Popular Products')
pdf.chapter_body(f"Top 10 Popular Products by Quantity Sold:\n{top_products.to_string()}\n")

pdf.chapter_title('Business Insight 3: Monthly Sales Trend')
pdf.chapter_body(f"Monthly Sales Trend:\n{monthly_sales.to_string()}\n")

pdf.chapter_title('Business Insight 4: Average Order Value by Customer')
pdf.chapter_body(f"Average Order Value by Customer:\n{avg_order_value}\n")

pdf.chapter_title('Business Insight 5: Product Category Revenue Contribution')
pdf.chapter_body(f"Revenue by Product Category:\n{category_revenue.to_string()}\n")

# Save PDF to file
pdf.output('Business_Insights_Report.pdf')
print("\nPDF report generated successfully!")