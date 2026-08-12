def divide_numbers(num1, num2):
    """
    Divide two numbers and return the result.
    
    Parameters:
    num1 (float): The numerator.
    num2 (float): The denominator.

    Returns:
    float: The result of the division.
    """
    if num2 == 0:
        raise ValueError("Cannot divide by zero.")
    return num1 / num2