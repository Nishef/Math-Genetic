import tkinter as tk
from tkinter import Menu
from sympy import symbols, diff, simplify

def solve_derivative(event=None):
    equation = entry.get()

    if not equation:
        result_label.config(text="Please enter your equation.")
        return

    # Define the variable and the equation
    x = symbols('x')
    y = simplify(equation)

    # Find the derivative
    derivative = diff(y, x)

    # Display the result
    result_label.config(text=f"Derivative: {derivative}")

def copy_to_clipboard(event):
    entry.event_generate('<<Copy>>')

def cut_to_clipboard(event):
    entry.event_generate('<<Cut>>')

def paste_from_clipboard(event):
    entry.event_generate('<<Paste>>')
    return "break"  # Prevent the default paste action

# Create the main window
window = tk.Tk()
window.title("Derivative Solver")
window.geometry("400x250")

# Create and place widgets
label = tk.Label(window, text="Enter the equation (use 'x' as the variable):", font=("Arial", 12))
label.pack(pady=10)

entry = tk.Entry(window, width=30, font=("Arial", 10))
entry.pack(pady=10)

# Add context menu for right-click
context_menu = Menu(window, tearoff=0)
context_menu.add_command(label="Copy", command=lambda: entry.event_generate('<<Copy>>'))
context_menu.add_command(label="Cut", command=lambda: entry.event_generate('<<Cut>>'))
context_menu.add_command(label="Paste", command=lambda: entry.event_generate('<<Paste>>'))
entry.bind("<Button-3>", lambda event: context_menu.post(event.x_root, event.y_root))

# Bind Ctrl+C, Ctrl+X, and Ctrl+V to copy, cut, and paste
entry.bind("<Control-c>", copy_to_clipboard)
entry.bind("<Control-x>", cut_to_clipboard)
entry.bind("<Control-v>", paste_from_clipboard)

# Bind Enter key to solve_derivative
entry.bind("<Return>", solve_derivative)

solve_button = tk.Button(window, text="Solve Derivative", command=solve_derivative, font=("Arial", 12))
solve_button.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()
