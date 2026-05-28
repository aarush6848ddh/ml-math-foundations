import numpy as np

# Part 1: Numerical dot product

v1 = np.array([1, 2])
v2 = np.array([4, -5])

dot_prod = v1[0] * v2[0] + v1[1] * v2[1]

numpy_dot = np.dot(v1, v2)

print(f"Manual: {dot_prod}")
print(f"NumPy: {numpy_dot}")


# Part 2: Geometric interpretation

# similar direction, dot product should be positive
v1 = np.array([1, 2])
v2 = np.array([2, 3])
print(f"Similar direction: {np.dot(v1, v2)}")

# perpendicular, dot product should be zero
v3 = np.array([3, 4])
v4 = np.array([4, -3])
print(f"Perpendicular: {np.dot(v3, v4)}")

# opposing direction, dot product should be negative
v5 = np.array([1, 2])
v6 = np.array([-2, -9])
print(f"Opposing Directions: {np.dot(v5, v6)}")


# Part 3: Projection

# take any vector and a unit vector
v = np.array([3, 4])
u = np.array([1, 0])  # unit vector pointing along x-axis

# a unit vector has length 1 
print(f"Length of u: {np.linalg.norm(u)}")

# project v onto u using the dot product
# the dot product with a unit vector gives you the projection length directly
projection_length = np.dot(v, u)
print(f"Projection of v onto u: {projection_length}")

# the actual projection vector is the length times the unit vector
projection_vector = projection_length * u
print(f"Projection vector: {projection_vector}")




# Connection to AI/ML:

# the dot product measures how much two vectors point in the same direction
# this is exactly what attention does in transformers
# it asks "how relevant is this word to that word"
# and answers it by computing a dot product between their vector representations
# high dot product means highly relevant, low or negative means not relevant
# that is the entire intuition behind attention