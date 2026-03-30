# user_interface.py

import tkinter as tk
from tkinter import filedialog
import subprocess
import sys

def select_file():
    file_path = filedialog.askopenfilename()
    entry_file.delete(0, tk.END)
    entry_file.insert(0, file_path)

def run_pipeline():
    file_path = entry_file.get()
    lookback = entry_lookback.get()
    train_start = entry_train_start.get()
    train_end = entry_train_end.get()
    test_start = entry_test_start.get()
    test_end = entry_test_end.get()

    subprocess.run([
        sys.executable, "run_pipeline.py",
        file_path,
        lookback,
        train_start,
        train_end,
        test_start,
        test_end
    ])

root = tk.Tk()
root.title("Futures ML Pipeline GUI")

tk.Label(root, text="Select Data File").pack()
entry_file = tk.Entry(root, width=50)
entry_file.pack()
tk.Button(root, text="Browse", command=select_file).pack()

tk.Label(root, text="Lookback Window").pack()
entry_lookback = tk.Entry(root)
entry_lookback.pack()

tk.Label(root, text="Train Start Date (YYYY-MM-DD)").pack()
entry_train_start = tk.Entry(root)
entry_train_start.pack()

tk.Label(root, text="Train End Date (YYYY-MM-DD)").pack()
entry_train_end = tk.Entry(root)
entry_train_end.pack()

tk.Label(root, text="Test Start Date (YYYY-MM-DD)").pack()
entry_test_start = tk.Entry(root)
entry_test_start.pack()

tk.Label(root, text="Test End Date (YYYY-MM-DD)").pack()
entry_test_end = tk.Entry(root)
entry_test_end.pack()

tk.Button(root, text="Run Pipeline", command=run_pipeline).pack()

root.mainloop()
