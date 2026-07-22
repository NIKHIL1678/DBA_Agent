"""
Seeds the dummy e-commerce database with realistic bulk data so the DBA
agents have something meaningful to query, analyze, and optimize.

Usage:
    pip install faker sqlalchemy pymysql
    # Set DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME env vars
    # (or edit the defaults in Database/connection.py)
    # Run schema.sql against your MySQL instance FIRST, then:
    python database/seed_data.py

Scale (adjust the constants below if you want it bigger/smaller):
    5,000   customers
    15      categories
    300     products
    20,000  orders
    ~50,000 order_items
"""

import random
from datetime import datetime, timedelta

from utils.Logging_Config import setup_logging

setup_logging()

from faker import Faker
from sqlalchemy import text
from Database.connection import get_db_engine

fake = Faker()
Faker.seed(42)
random.seed(42)

NUM_CUSTOMERS = 5_000
CATEGORY_NAMES = [
    "Electronics", "Books", "Home & Kitchen", "Clothing", "Toys",
    "Sports & Outdoors", "Beauty", "Automotive", "Garden", "Office Supplies",
    "Pet Supplies", "Health", "Grocery", "Music", "Tools",
]
NUM_PRODUCTS = 300
NUM_ORDERS = 20_000
ORDER_STATUSES = ["pending", "shipped", "delivered", "cancelled", "returned"]
BATCH_SIZE = 1000


def seed_customers(engine):
    print(f"Seeding {NUM_CUSTOMERS} customers...")
    rows = []
    for _ in range(NUM_CUSTOMERS):
        rows.append({
            "full_name": fake.name(),
            "email": fake.email(),
            "city": fake.city(),
            "country": fake.country(),
            "created_at": fake.date_time_between(start_date="-3y", end_date="now"),
        })

    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            conn.execute(
                text("""
                    INSERT INTO customers (full_name, email, city, country, created_at)
                    VALUES (:full_name, :email, :city, :country, :created_at)
                """),
                batch,
            )
    print("  done.")


def seed_categories(engine):
    print(f"Seeding {len(CATEGORY_NAMES)} categories...")
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO categories (name) VALUES (:name)"),
            [{"name": name} for name in CATEGORY_NAMES],
        )
    print("  done.")


def seed_products(engine):
    print(f"Seeding {NUM_PRODUCTS} products...")
    with engine.connect() as conn:
        category_ids = [row[0] for row in conn.execute(text("SELECT id FROM categories")).fetchall()]

    rows = []
    for _ in range(NUM_PRODUCTS):
        rows.append({
            "name": fake.unique.catch_phrase(),
            "category_id": random.choice(category_ids),
            "price": round(random.uniform(3.99, 799.99), 2),
            "stock": random.randint(0, 500),
        })

    with engine.begin() as conn:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            conn.execute(
                text("""
                    INSERT INTO products (name, category_id, price, stock)
                    VALUES (:name, :category_id, :price, :stock)
                """),
                batch,
            )
    print("  done.")


def seed_orders_and_items(engine):
    print(f"Seeding {NUM_ORDERS} orders (plus order_items)...")
    with engine.connect() as conn:
        customer_ids = [row[0] for row in conn.execute(text("SELECT id FROM customers")).fetchall()]
        product_rows = conn.execute(text("SELECT id, price FROM products")).fetchall()

    product_ids = [row[0] for row in product_rows]
    product_prices = {row[0]: float(row[1]) for row in product_rows}

    order_rows = []
    for _ in range(NUM_ORDERS):
        order_date = fake.date_time_between(start_date="-2y", end_date="now")
        order_rows.append({
            "customer_id": random.choice(customer_ids),
            "order_date": order_date,
            "status": random.choice(ORDER_STATUSES),
            "total": 0.0,  # filled in after items are generated below
        })

    with engine.begin() as conn:
        for i in range(0, len(order_rows), BATCH_SIZE):
            batch = order_rows[i:i + BATCH_SIZE]
            conn.execute(
                text("""
                    INSERT INTO orders (customer_id, order_date, status, total)
                    VALUES (:customer_id, :order_date, :status, :total)
                """),
                batch,
            )

    with engine.connect() as conn:
        order_ids = [row[0] for row in conn.execute(text("SELECT id FROM orders")).fetchall()]

    print(f"Seeding order_items for {len(order_ids)} orders...")
    item_rows = []
    order_totals = {}  # order_id -> running total, applied back to orders after

    for order_id in order_ids:
        num_items = random.randint(1, 5)
        running_total = 0.0
        chosen_products = random.sample(product_ids, k=min(num_items, len(product_ids)))
        for product_id in chosen_products:
            quantity = random.randint(1, 4)
            unit_price = product_prices[product_id]
            running_total += quantity * unit_price
            item_rows.append({
                "order_id": order_id,
                "product_id": product_id,
                "quantity": quantity,
                "unit_price": unit_price,
            })
        order_totals[order_id] = round(running_total, 2)

    with engine.begin() as conn:
        for i in range(0, len(item_rows), BATCH_SIZE):
            batch = item_rows[i:i + BATCH_SIZE]
            conn.execute(
                text("""
                    INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                    VALUES (:order_id, :product_id, :quantity, :unit_price)
                """),
                batch,
            )

    print("Backfilling order totals...")
    total_updates = [{"id": oid, "total": total} for oid, total in order_totals.items()]
    with engine.begin() as conn:
        for i in range(0, len(total_updates), BATCH_SIZE):
            batch = total_updates[i:i + BATCH_SIZE]
            conn.execute(
                text("UPDATE orders SET total = :total WHERE id = :id"),
                batch,
            )
    print("  done.")


def main():
    engine = get_db_engine()
    seed_customers(engine)
    seed_categories(engine)
    seed_products(engine)
    seed_orders_and_items(engine)
    print("\nSeeding complete: 5,000 customers / 15 categories / 300 products / "
          "20,000 orders / ~60,000 order_items.")


if __name__ == "__main__":
    main()