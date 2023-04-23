# %%

import gnp

a = gnp.array([9.0, 18.0, 27.0])
b = gnp.array([3.0, 4.0, 5.0])
c = gnp.array(4.25)
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")

# %%

# Unary ops
print(f"-a = {-a}")
print(f"+a = {+a}")
print(f"~a = {~a}")

print(f"-b = {-b}")
print(f"+b = {+b}")
print(f"~b = {~b}")

print(f"-c = {-c}")
print(f"+c = {+c}")
print(f"~c = {~c}")

# %%

# Binary ops

# addition testing
print(f"a + b = {a + b}")
print(f"a + c = {a + c}")

# subtraction testing
print(f"a - b = {a - b}")
print(f"a - c = {a - c}")

# multiplication testing
print(f"a * b = {a * b}")
print(f"a * c = {a * c}")

# True divide
print(f"a / b = {a / b}")
print(f"a / c = {a / c}")

# Floor divide
print(f"a // b = {a // b}")
print(f"a // c = {a // c}")

# Mod 
print(f"a % b = {a % b}")
print(f"a % c = {a % c}")

# Pow
print(f"a ** b = {a ** b}")
print(f"a ** c = {a ** c}")


# %%

# Compare
print(f"a > b = {a > b}")
print(f"b >= c = {b >= c}")


# %%

# Left and right shift
d = gnp.array([1, 2, 4])
print(f"d >> 2 = {d >> gnp.array(2)}")
print(f"d << 4 = {d << gnp.array(4)}")

# %%

d <<= gnp.array(4)
print(f"d <<= 4. d = {d}")

# %%

d >>= gnp.array(2)
print(f"d >>= 2. d = {d}")

# %%

# mat mul
e = gnp.array([[1, 2, 3], [3, 4, 5]]) # (2, 3)
f = gnp.array([[0], [1], [2]]) # (3, 1)

print(f"e @ f = {e @ f}")

# %%


# batch mat mul
e = gnp.array([[[1, 2, 3], [3, 4, 5]]]) # (1, 2, 3)
f = gnp.array([[[0], [1], [2]]]) # (1, 3, 1)
print(f"e @ f = {e @ f}")

# %%



