import pyodbc
from dotenv import load_dotenv
import os
load_dotenv()
server = os.getenv("server")
database = os.getenv("database")
username="ChatAgent"
password = os.getenv("password")
driver = os.getenv("driver")   
tables = [
    "iivf_fact_transactions",
    "IIVF_FACT_PAYMENTS",
    "IIVF_Master_BillingServiceMainGroup",
    "IIVF_Fact_Payments"
]

conn = pyodbc.connect(
    f"DRIVER={driver};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=30;"
)

cursor = conn.cursor()

print("✅ Connected to Azure SQL using SQL Authentication\n")

for table in tables:
    try:
        cursor.execute(f"SELECT 1 FROM {table}")
        cursor.fetchone()
        print(f"✅ Table accessible: {table}")
    except Exception as e:
        print(f"❌ Table access failed: {table}\n   {e}")

cursor.close()
conn.close()

print("\n🔒 Connection closed successfully")
