import numpy as np


# Part 1: Composition of two transformations

# rotation matrix
rotation_matrix = np.array([
    [0, -1],
    [1, 0]
])

# shear matrix
shear_matrix = np.array([
    [1, 1],
    [0, 1]
])

# vector to transform
v = np.array([2, 1])

# apply rotation first then shear the long way
step1 = np.matmul(rotation_matrix, v) # rotate v
step2 = np.matmul(shear_matrix, step1) # shear the result of step1
print(f"Long way: {step2}")

# compute the composition matrix in one shot
composition = np.matmul(shear_matrix, rotation_matrix) # np.matmul(shear_matrix, rotation_matrix)
result = np.matmul(composition, v) # apply composition to v
print(f"Composition: {result}")


# Part 2: Order matters (non-commutativity)

# we could show that matrix multiplication isn't commutative through numerical computation
rotation_shear = np.matmul(rotation_matrix, shear_matrix)
shear_rotation = np.matmul(shear_matrix, rotation_matrix)
print(f"rotation @ shear: {rotation_shear},\nshear @ rotation: {shear_rotation}")


# Part 3: The exception: scaling matrix commutes with everything

# identity matrix scaled by 5
scaling_matrix = np.array([
    [5, 0],
    [0, 5]
])

normal_matrix = np.array([
    [1, 5],
    [3, 5]
])

composition1 = np.matmul(scaling_matrix, normal_matrix)
composition2 = np.matmul(normal_matrix, scaling_matrix)
print(f"scaling @ normal: {composition1},\nnormal @ scaling: {composition2}")
print(f"Do they commute? {np.array_equal(composition1, composition2)}")
# A scaling matrix is just the identity matrix times a scalar (5).
# The identity does nothing, so it commutes with everything.
# Scaling uniformly in all directions does not interact with
# any other transformation, so order does not matter.


# Part 4: Associativity

a = np.array([
    [3, 4],
    [5, 7]
])

b = np.array([
    [5, 8],
    [3, 2]
])

c = np.array([
    [9, 6],
    [7, 1]
])

computation1 = np.matmul((np.matmul(a, b)), c)
computation2 = np.matmul(a, (np.matmul(b, c)))
print(f"Are they the same? {np.array_equal(computation1, computation2)}")


# Connection to AI/ML:

# Every forward pass through a neural network is matrix composition.
# Each layer applies a transformation and the result flows into the next layer.
# The order of layers is non-negotiable, just like matrix order matters.
# You cannot swap attention and feedforward and expect the same result.

# Associativity means you can group matrix multiplications however you want
# as long as the order stays the same.
# This is why GPUs can batch and parallelize matrix operations efficiently
# without changing the final result.

# When you see x @ W1 @ W2 @ W3 in a transformer
# it does not matter if you compute (x @ W1 @ W2) @ W3
# or x @ (W1 @ W2 @ W3), the result is identical.
# This is associativity making deep networks computationally efficient.