def explist(low, high, n):
    n = 100
    low = 0.0001
    high = 0.1

    ratio = high / low

    # i.e. low * ratio = high
    # Find out the "x" which raised to the power of "n" gives "ratio":
    x = ratio ** (1 / n)

    # i.e. low * x**n = high
    # "n" exponentially distributed numbers between "low" and "high":
    return list(map(lambda i: low * x**i, range(100)))

