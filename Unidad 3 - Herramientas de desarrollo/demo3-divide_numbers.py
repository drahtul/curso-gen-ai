def divide_numbers(num1, num2):
    # Esta funcion encapsula una operacion que podria ser utilizada como
    # herramienta por un agente: recibe entradas, valida un caso limite y
    # devuelve un resultado determinista.
    """
    Divide two numbers and return the result.
    
    Parameters:
    num1 (float): The numerator.
    num2 (float): The denominator.

    Returns:
    float: The result of the division.
    """
    # La validacion evita una operacion indefinida y convierte el problema en
    # un error explicito que la aplicacion llamadora puede manejar.
    if num2 == 0:
        raise ValueError("Cannot divide by zero.")
    return num1 / num2