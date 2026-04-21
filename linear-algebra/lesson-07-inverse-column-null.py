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

# full rank matrix: columns are linearly independent, spans full 2D space
full_rank = np.array([
    [1, 2],
    [3, 4]
])

# rank deficient matrix: columns are linearly dependent, collapses to a line
rank_deficient = np.array([
    [1, 2],
    [2, 4]
])

# compute ranks
rank1 = np.linalg.matrix_rank(full_rank)
rank2 = np.linalg.matrix_rank(rank_deficient)

print(f"Full rank matrix rank: {rank1}")       # should be 2
print(f"Rank deficient matrix rank: {rank2}")  # should be 1

# rank 2 means the transformation spans all of 2D space
# rank 1 means the transformation squishes everything onto a single line


# Part 3: Null Space

# the null space is all vectors that map to zero
# for [[1,2],[2,4]] the vector [2,-1] maps to zero
# because 2*col1 - 1*col2 = [2,4] - [2,4] = [0,0]

null_vector = np.array([2, -1])

result = np.matmul(rank_deficient, null_vector)
print(f"rank_deficient @ [2,-1] = {result}")
# should print [0, 0]

# this means [2,-1] lives in the null space
# the transformation completely squishes it to the origin


# Part 4: Irreversibility

# a singular matrix has determinant 0 and no inverse
# trying to invert it should fail
singular_matrix = np.array([
    [1, 2],
    [2, 4]
])

try:
    inverse = np.linalg.inv(singular_matrix)
    print(f"Inverse: {inverse}")
except np.linalg.LinAlgError as e:
    print(f"Cannot invert: {e}")

# you cannot unsquish a line back into a plane
# once information is destroyed by a rank deficient transformation
# it is gone forever and cannot be recovered


# Connection to AI/ML:


# rank deficient weight matrices in neural networks destroy information permanently
# just like a singular matrix has no inverse, a collapsed layer cannot be recovered
# this is why full rank weight matrices are important for network capacity

# solving Ax = b is exactly what a neural network does during inference
# the network learns the matrix A during training
# and at inference time solves for the output given the input

# the null space of a weight matrix represents features the layer ignores completely
# if important features fall in the null space, the network can never learn them
# this is why architecture design and initialization matter so much
