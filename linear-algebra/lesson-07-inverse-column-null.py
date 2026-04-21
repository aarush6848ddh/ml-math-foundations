import numpy as np


# Part 1:  Solving a Linear System

matrix_a = np.array([
    [1, 4],
    [3, 5]
])

vector_b = np.array([1, 7])

# calculate the inverse matrix
a_inv = np.linalg.inv(matrix_a)

# solve for x using the inverse 
x = np.matmul(a_inv, vector_b)

# verify by checking A @ x gives back b
verification = np.matmul(matrix_a, x) # matrix_a @ x

print(f"Solution x: {x}")
print(f"Verification A @ x: {verification}")
print(f"Matches b: {np.allclose(verification, vector_b)}")


# Part 2: Column Space and Rank




# Part 3: The exception: scaling matrix commutes with everything




# Part 4: Associativity




# Connection to AI/ML:

