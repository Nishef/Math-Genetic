from sympy import symbols, simplify, sympify, Add, Mul, Pow, diff ,Abs
import tkinter as tk

def replace_real_numbers(equation):
    x = symbols('x')
    equation = sympify(equation)  # Ensure the equation is a Sympy expression

    # Replace real numbers in the coefficients with W_n
    def replace_coefficients(expr):
        nonlocal w_count
        if expr.is_real:
            result = symbols(f'W_{w_count}')
            w_count += 1
            return result
        elif isinstance(expr, Add):
            return expr.func(*[replace_coefficients(term) for term in expr.args])
        elif isinstance(expr, Mul):
            args = [replace_coefficients(factor) for factor in expr.args]
            return expr.func(*args)
        elif isinstance(expr, Pow) and expr.base == x:
            return expr.func(replace_coefficients(expr.base), expr.exp)
        else:
            return expr

    w_count = 0
    result = replace_coefficients(equation)

    return result

def calculate_derivative(equation):
    x = symbols('x')
    y = simplify(equation)
    derivative = diff(y, x)
    return derivative

def find_x_term(main_equation):
    x = symbols('x')
    # Extract the coefficients of the terms in the main equation
    main_equation_dict = sympify(main_equation).as_coefficients_dict()
    # Find the term involving 'x'
    x_term = next((term for term in main_equation_dict if x in term.free_symbols), None)
    return x_term, main_equation_dict.get(x_term, 0)

def solve_fitness(main_equation_entry, exact_equation_entry, initial_value_entry, result_label, replaced_label):
    main_equation = main_equation_entry.get()
    exact_equation = exact_equation_entry.get()
    initial_x_value = sympify(initial_value_entry.get())  # Convert to symbolic expression

    W_0 = symbols("W_0")
    initial_x = Abs(W_0 - initial_x_value)

    # Simplify the expression
    simplified_initial_x = simplify(initial_x)

    # Extract the arguments of Abs() and display them
    if isinstance(simplified_initial_x, Abs):
        inside_abs = simplified_initial_x.args[0]
        print(f"Simplified initial_x: {inside_abs}")

    if not main_equation or not exact_equation:
        result_label.config(text="Please enter both main and exact equations.")
        return
    else:
        result_label.config(text="Your fitness is:")

    # Automatically find the term involving 'x'
    x_term, coefficient_of_x = find_x_term(main_equation)

    if x_term:
        # Save the term involving 'x' and its coefficient for later use
        print(f"Term involving 'x': {x_term}")
        print(f"Coefficient of {x_term}: {coefficient_of_x}")
        real_x = x_term * coefficient_of_x
        main_equation = sympify(main_equation).subs("x",0)

    while "y" in str(main_equation) or "d" in str(main_equation):

        if "y" in str(main_equation) and "d" in str(main_equation):

            replaced_y_exact_equation = replace_real_numbers(exact_equation)
            replaced_y_exact_equation = sympify(main_equation).subs("y", replaced_y_exact_equation)
            derivative_replaced_y_exact_equation = calculate_derivative(replaced_y_exact_equation)
            full_new = replaced_y_exact_equation.subs("d", derivative_replaced_y_exact_equation)
            if x_term:
                full_new = sympify(full_new+real_x+inside_abs)

            replaced_label.delete("1.0", tk.END)
            replaced_label.insert(tk.END, f"{full_new}")

        elif "d" in str(main_equation) and not "y" in str(main_equation):
            # Replace 'y'' with the derivative of exact_equation
            replaced_dy_exact_equation = sympify(main_equation).subs("d", exact_equation)
            replaced_dy_exact_equation = simplify(replaced_dy_exact_equation)
            replaced_exact_equation = replace_real_numbers(replaced_dy_exact_equation)
            replaced_exact_equation = simplify(replaced_exact_equation)
            derivative_replaced_exact_equation = calculate_derivative(replaced_exact_equation)
            derivative_replaced_exact_equation = sympify(main_equation).subs("d", derivative_replaced_exact_equation)

            if x_term:
                derivative_replaced_exact_equation = sympify(derivative_replaced_exact_equation+real_x+inside_abs)

            replaced_label.delete("1.0", tk.END)
            replaced_label.insert(tk.END, f"{derivative_replaced_exact_equation}")

        elif "y" in str(main_equation) and not "d" in str(main_equation):
            # Replace 'y' in the main equation with the exact equation after running replace_real_numbers
            replaced_y_exact_equation = sympify(main_equation).subs("y", exact_equation)
            replaced_y_exact_equation = replace_real_numbers(replaced_y_exact_equation)

            if x_term:
                replaced_y_exact_equation= sympify(replaced_y_exact_equation+real_x+inside_abs)

            replaced_label.delete("1.0", tk.END)
            replaced_label.insert(tk.END, f"{replaced_y_exact_equation}")

        break

    print(f"Original Main Equation: {main_equation}")