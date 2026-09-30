import tkinter as tk
import pandas as pd
from tkinter import ttk, messagebox
from predict import load_model, prepare_data, predict_sales

model, columns = load_model()
df = prepare_data()

def predict():
    date = date_entry.get()
    store = store_entry.get()
    family = family_combobox.get()
    promotion = promotion_combobox.get()

    if not date or not store or not family or not promotion:
        messagebox.showwarning(
            "Missing information", 
            "Please fill in all fields."
        )
        return

    try:
        store = int(store)

        date = pd.to_datetime(date)

        prediction = predict_sales(
            model, 
            columns,
            df, 
            date, 
            store,
            family, 
            promotion
        )

        result_label.config(
            text=f"Predicted Sales: {prediction:.2f}"
        )
    except ValueError as error:
        messagebox.showerror(
            "Invalid imput",
            str(error)
        )


# Create the main window
window = tk.Tk()

window.title("Store Sales Forecast")
window.geometry("500x500")
window.resizable(False, False)

# Title
title_label = ttk.Label(
    window,
    text="Store Sales Forecast",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=30)

# Date
date_label = ttk.Label(
    window,
    text="Date:"
)

date_label.pack()

date_entry = ttk.Entry(
    window,
    width=30
)

date_entry.pack(pady=(5, 15))

#Store
store_label = ttk.Label(
    window,
    text="Store:"
)

store_label.pack()

store_entry = ttk.Entry(
    window,
    width=30
)

store_entry.pack(pady=(5, 15))

# Product Family
family_label = ttk.Label(
    window,
    text="Product Family:"
)

family_label.pack()

family_combobox = ttk.Combobox(
    window,
    values=[
        "AUTOMOTIVE",
        "BEAUTY",
        "BEVERAGES",
        "BOOKS",
        "BREAD/BAKERY",
        "CELEBRATION",
        "CLEANING",
        "DAIRY",
        "DELI",
        "EGGS",
        "FROZEN FOODS",
        "GROCERY I",
        "GROCERY II",
        "HARDWARE",
        "HOME AND KITCHEN I",
        "HOME AND KITCHEN II",
        "HOME APPLIANCES",
        "HOME CARE",
        "LADIESWEAR",
        "LAWN AND GARDEN",
        "LINGERIE",
        "LIQUOR,WINE,BEER",
        "MAGAZINES",
        "MEATS",
        "PERSONAL CARE",
        "PET SUPPLIES",
        "PLAYERS AND ELECTRONICS",
        "POULTRY",
        "PREPARED FOODS",
        "PRODUCE",
        "SCHOOL AND OFFICE SUPPLIES",
        "SEAFOOD"
    ],
    state="readonly",
    width=27
)

family_combobox.pack(pady=(5, 15))

# Promotion
promotion_label = ttk.Label(
    window,
    text="On Promotion:"
)

promotion_label.pack()

promotion_combobox = ttk.Combobox(
    window,
    values=["Yes", "No"],
    state="readonly",
    width=27
)

promotion_combobox.pack(pady=(5, 25))

# Predict Button
predict_button = ttk.Button(
    window, 
    text="Predict Sales",
    command=predict
)

predict_button.pack(pady=10)

# Result
result_label = ttk.Label(
    window,
    text="Predicted Sales: --",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=25)

# Start the application

window.mainloop()