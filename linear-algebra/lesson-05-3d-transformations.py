import numpy as np


# Part 1: 3D matrix vector multiplication

# 3D matrix
matrix = np.array([
    [0, 0.5, -0.5],
    [0, 0.5, 1],
    [1, 0, 0.5]
])

# 3D vector
v = np.array([1, 0, -2])

# resulting output vector
result = np.matmul(matrix, v)

print(f"Resulting output vector: {result}")


# Part 2: 3D rotation matrix

# 90 degree counterclockwise rotation around the z-axis
# the columns tell you where each basis vector lands
rotation_3d = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])

# verify each basis vector lands where the lesson says it should
i_hat = np.array([1, 0, 0])
j_hat = np.array([0, 1, 0])
k_hat = np.array([0, 0, 1])


print(f"i_hat lands at: {np.matmul(rotation_3d, i_hat)}")
print(f"j_hat lands at: {np.matmul(rotation_3d, j_hat)}")
print(f"k_hat lands at: {np.matmul(rotation_3d, k_hat)}")



# Connection to AI/ML:

# Everything we did in 2D and 3D works identically in any number of dimensions.
# GPT-4 uses 12,288 dimensional vectors for each token.
# Every attention operation and weight matrix in a transformer is just
# this same operation, matrix vector multiplication, in very high dimensions.
# The dimension changes, the concept does not.
# When you see embedding_size=512 in a transformer config,
# that is a 512 dimensional version of exactly what we just coded.