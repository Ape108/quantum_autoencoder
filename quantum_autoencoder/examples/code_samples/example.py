def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def main():
    # Test Fibonacci
    print("Fibonacci sequence:")
    for i in range(10):
        print(f"F({i}) = {fibonacci(i)}")
    
    # Test prime numbers
    print("\nPrime numbers up to 20:")
    for i in range(1, 21):
        if is_prime(i):
            print(f"{i} is prime")

if __name__ == "__main__":
    main() 