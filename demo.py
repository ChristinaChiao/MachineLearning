def multiplication_table():
    print("九九乘法表:")
    for i in range(1, 10):
        line = ""
        for j in range(1, 10):
            product = i * j
            line += f"{product:2}"
        print(line)

multiplication_table()