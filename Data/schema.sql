-- =====================================================================
-- Dummy e-commerce database for testing the DBA multi-agent system.
--
-- INTENTIONAL PERFORMANCE ISSUES (for the Query Analyst agent to find):
--   1. customers.email  -> no index, despite being a common lookup field
--   2. orders.status    -> no index, despite being a common filter field
--   3. orders.order_date -> no index, despite being used for date-range reports
--   4. order_items      -> no composite index on (order_id, product_id),
--                           only single-column FKs
--
-- Everything else (primary keys, foreign keys) is indexed normally, so the
-- agent has to actually reason about which specific columns are missing
-- coverage rather than finding an obviously broken table.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS dba_agent_test
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE dba_agent_test;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS customers;

-- ---------------------------------------------------------------------
CREATE TABLE customers (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    full_name   VARCHAR(120)  NOT NULL,
    email       VARCHAR(150)  NOT NULL,      -- deliberately NOT indexed/unique
    city        VARCHAR(80)   NOT NULL,
    country     VARCHAR(80)   NOT NULL,
    created_at  DATETIME      NOT NULL
);

-- ---------------------------------------------------------------------
CREATE TABLE categories (
    id      INT AUTO_INCREMENT PRIMARY KEY,
    name    VARCHAR(80) NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------
CREATE TABLE products (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(150)   NOT NULL,
    category_id INT            NOT NULL,
    price       DECIMAL(10, 2) NOT NULL,
    stock       INT            NOT NULL DEFAULT 0,
    CONSTRAINT fk_products_category
        FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- ---------------------------------------------------------------------
CREATE TABLE orders (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT            NOT NULL,
    order_date  DATETIME       NOT NULL,     -- deliberately NOT indexed
    status      VARCHAR(20)    NOT NULL,     -- deliberately NOT indexed
    total       DECIMAL(10, 2) NOT NULL,
    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    -- Note: FK creates an index on customer_id automatically in MySQL/InnoDB,
    -- so customer_id lookups will already be reasonably fast. status and
    -- order_date are the two blind spots to leave for the agent to catch.
);

-- ---------------------------------------------------------------------
CREATE TABLE order_items (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    order_id    INT            NOT NULL,
    product_id  INT            NOT NULL,
    quantity    INT            NOT NULL,
    unit_price  DECIMAL(10, 2) NOT NULL,
    CONSTRAINT fk_items_order
        FOREIGN KEY (order_id) REFERENCES orders(id),
    CONSTRAINT fk_items_product
        FOREIGN KEY (product_id) REFERENCES products(id)
    -- No composite index on (order_id, product_id) -- a report that joins
    -- and filters on both will do more work than it needs to.
);