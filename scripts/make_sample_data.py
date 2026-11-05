"""Generate reproducible sample datasets (with a few planted anomalies and weak products)."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out = Path(__file__).resolve().parent.parent / "sample_data"
out.mkdir(exist_ok=True)

regions = ["North", "South", "East", "West", "Central"]
region_w = [0.24, 0.18, 0.27, 0.21, 0.10]
products = {
    "Laptop Pro": ("Electronics", 1200, 5), "Wireless Mouse": ("Accessories", 25, 14),
    "Mechanical Keyboard": ("Accessories", 90, 9), "4K Monitor": ("Electronics", 380, 6),
    "USB-C Hub": ("Accessories", 45, 10), "Noise-Cancelling Headphones": ("Audio", 220, 7),
    "Bluetooth Speaker": ("Audio", 80, 6), "Webcam HD": ("Electronics", 70, 4),
    "Desk Lamp": ("Office", 35, 5), "Standing Desk": ("Office", 450, 2),   # underperformers
    "Ergonomic Chair": ("Office", 300, 2), "Laptop Stand": ("Office", 40, 1),
}
names = list(products)
prod_w = np.array([products[p][2] for p in names], float)
prod_w /= prod_w.sum()

# customers
n_cust = 60
first = ["Aarav", "Liam", "Noah", "Emma", "Olivia", "Mia", "Zara", "Ethan", "Sofia", "Lucas", "Priya", "Omar"]
last = ["Patel", "Smith", "Khan", "Johnson", "Singh", "Garcia", "Brown", "Ali", "Williams", "Sharma"]
customers = pd.DataFrame({
    "customer_id": [f"C{i:03d}" for i in range(1, n_cust + 1)],
    "customer_name": [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n_cust)],
    "segment": rng.choice(["Enterprise", "SMB", "Consumer"], n_cust, p=[0.2, 0.35, 0.45]),
    "country": rng.choice(["India", "UK", "USA", "Germany"], n_cust, p=[0.3, 0.3, 0.25, 0.15]),
    "signup_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_cust), unit="D"),
})
customers["customer_name"] = customers["customer_name"] + customers.groupby("customer_name").cumcount().map(lambda i: "" if i == 0 else f" {i+1}")

# sales
n = 1800
dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
month_boost = pd.Series(dates.month).map(lambda m: 1.35 if m in (11, 12) else 1.0).values
keep = rng.random(n) < (month_boost / 1.35)
dates = dates[keep]
n = len(dates)
prod = rng.choice(names, n, p=prod_w)
price = np.array([products[p][1] for p in prod]) * rng.uniform(0.95, 1.05, n)
qty = rng.integers(1, 6, n)
disc = rng.choice([0, 0, 0, 0.05, 0.1, 0.15], n)
sales = pd.DataFrame({
    "order_id": [f"ORD{i:05d}" for i in range(1, n + 1)],
    "order_date": dates.strftime("%Y-%m-%d"),
    "region": rng.choice(regions, n, p=region_w),
    "product": prod,
    "category": [products[p][0] for p in prod],
    "customer_id": rng.choice(customers["customer_id"], n),
    "quantity": qty,
    "unit_price": price.round(2),
    "discount": disc,
})
sales["revenue"] = (sales.quantity * sales.unit_price * (1 - sales.discount)).round(2)
sales = sales.sort_values("order_date").reset_index(drop=True)

# planted anomalies
sales.loc[100, ["quantity", "revenue"]] = [400, 400 * sales.loc[100, "unit_price"]]     # bulk-order typo
sales.loc[450, "revenue"] = sales.loc[450, "revenue"] * 25                             # inflated revenue
sales.loc[900, "revenue"] = -abs(sales.loc[900, "revenue"])                            # refund / negative sale
sales.loc[1300, "unit_price"] = 99999.0                                                # price entry error

sales.to_csv(out / "sales.csv", index=False)
customers.to_csv(out / "customers.csv", index=False)
print(f"sales={len(sales)} customers={len(customers)} -> {out}")
