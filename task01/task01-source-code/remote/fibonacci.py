# Fibonacci sequence generator

def fibonacci(number):
    """
    Fibonacci
    >>> fibonacci(0)
    [0]
    >>> fibonacci(-10)
    Error!
    >>> fibonacci(4)
    [0, 1, 1, 2, 3]
    """

    if number < 0:
        return print("Error!")

    A, B = 1, 0
    for i in range(number + 1):
        yield B
        A, B = B, A + B

if __name__ == "__main__":
    try:
        fibo_length = int(input("Please input the length of fibonacci sequence: "))
        for i in fibonacci(fibo_length):
            print(i)
    except Exception as e:
        print("Error Input:", e)