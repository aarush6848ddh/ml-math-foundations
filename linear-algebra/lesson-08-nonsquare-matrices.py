import numpy as np


# Part 1: 2D to 3D Transformation

three_by_two = np.array([
    [1, 0], 
    [3, 5],
    [9, 2]
])

v = np.array([1, 4])

t = np.matmul(three_by_two, v)

print(f"Input vector: {v}")
print(f"Input shape: {v.shape}")
print(f"Output vector: {t}")
print(f"Output shape: {t.shape}")

# Part 2: 3D to 2D transformation

two_by_three = np.array([
    [4, 6, 3],
    [7, 2, 9]
])

v2 = np.array([4, 6, 2])


t2 =  np.matmul(two_by_three, v2)

print(f"Input vector: {v2}")
print(f"Input shape: {v2.shape}")
print(f"Output vector: {t2}")
print(f"Output shape: {t2.shape}")

# Connection to AI/ML:

# this is exactly what happens inside every transformer
# the embedding layer takes a token ID which is just a single number
# and maps it up to a high dimensional vector like 512D or 1024D
# that is a nonsquare matrix mapping from low dimension to high dimension

# the output projection layer at the end of the transformer does the opposite
# it takes the high dimensional representation and maps it down to vocabulary size
# so the model can predict the next token
# that is a nonsquare matrix mapping from high dimension to low dimension

# every linear layer in a neural network is a nonsquare matrix transformation
# changing the dimensionality of the data as it flows through the network