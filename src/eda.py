import matplotlib.pyplot as plt
import pandas as pd

# print(df.head(), "\n", "-" * 90)

# print(df.info(), "\n", "-" * 90)

# print(df.describe(), "\n", "-" * 90)

# print(df.isnull(), "\n", "-" * 90)

# print(df.columns, "\n", "-" * 90)

# print(df.shape, "\n", "-" * 90)

def plot_sales_over_time(df):
    daily_sales = df.groupby("date")["sales"].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales.index, daily_sales.values)

    plt.title("Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sales_by_day_of_week(df):
    df = df.copy()

    df["day_of_week"] = df["date"].dt.day_name()

    order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]    

    sales_by_day = (
        df.groupby("day_of_week")["sales"].mean().reindex(order)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(sales_by_day.index, sales_by_day.values)

    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Sales")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sales_by_family(df):
    sales_by_family = (
        df.groupby("family")["sales"].mean().sort_values(ascending=True)
    )

    plt.figure(figsize=(12, 8))
    plt.barh(sales_by_family.index, sales_by_family.values)

    plt.title("Average Sales by Product Family")
    plt.xlabel("Average Sales")
    plt.ylabel("Product Family")

    plt.tight_layout()
    plt.show()


def plot_sales_by_store(df):
    sales_by_store = (
        df.groupby("store_nbr")["sales"].mean().sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    plt.bar(
        sales_by_store.index.astype(str),
        sales_by_store.values
    )

    plt.title("Average Sales by Store")
    plt.xlabel("Store")
    plt.ylabel("Average Sales")

    plt.tight_layout()
    plt.show()

def plot_promotion_vs_sales(df):
    df = df.copy()

    df["is_on_promotion"] = df["onpromotion"] > 0

    sales_by_promotion = (
        df.groupby("is_on_promotion")["sales"].mean()
    )

    plt.figure(figsize=(8, 6))
    plt.bar(
        ["No Promotion", "Promotion"], sales_by_promotion.values
    )

    plt.title("Average Sales: Promotion vs No Promotion")
    plt.xlabel("Promotion Status")
    plt.ylabel("Average Sales")

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------- # 

def plot_monthly_sales(df):
    df = df.copy()

    df["month"] = df["date"].dt.to_period("M")

    monthly_sales = (
        df.groupby("month")["sales"].sum()
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        monthly_sales.index.astype(str),
        monthly_sales.values,
        marker="o"
    )

    plt.title("Total Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------- #

def run_eda(df):
    plot_sales_over_time(df)
    plot_sales_by_day_of_week(df)
    plot_sales_by_family(df)
    plot_sales_by_store(df)
    plot_promotion_vs_sales(df)
    plot_monthly_sales(df)