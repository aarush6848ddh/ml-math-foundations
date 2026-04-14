import numpy as np

# Part 1: Apply a Transformation Matrix to a Vector
# column 1 = where i_hat [1, 0] lands
# column 2 = where j_hat [0, 1] lands
matrix = np.array([
    [1, 3],
    [2, 1]
])

# a vector to transform
v = np.array([5, 3])

# manual multiplication, scale column 1 by v[0] and scale column 2 by v[1]
# add them
col1 = matrix[:, 0]
col2 = matrix[:, 1]
manual_result = (v[0] * col1) + (v[1] * col2)
print(f"Manual result: {manual_result}")

# verify using numpy
numpy_result = np.matmul(matrix, v)
print(f"Numpy result: {numpy_result}")

# Part 2: Rotation Matrix (90 degrees counterclockwise)

# i_hat [1,0] lands at [0,1]
# j_hat [0,1] lands at [-1,0]
# so the rotation matrix columns are [0,1] and [-1,0]
rotation_matrix = np.array([
    [0, -1],
    [1, 0]
    ]
)

# apply a few vectors and show where they land 
test_vectors = [np.array([1, 0]), np.array([0, 1]), np.array([2, 3])]
for vec in test_vectors:
    result = (vec[0] * rotation_matrix[:, 0]) + (vec[1] * rotation_matrix[:, 1]) # np.matmul(rotation_matrix, vec)
    print(f"Rotation: {vec} -> {result}")

# Part 3: Shear Matrix 

# i_hat stays fixed at [1,0]
# j_hat moves to [1,1]
shear_matrix = np.array([
    [1, 1],
    [0, 1]
])

# apply to a few vectors
for vec in test_vectors:
    result = np.matmul(shear_matrix, vec)
    print(f"Shear: {vec} -> {result}")

# Part 4: Linearly Dependent Columns

dependent_matrix = np.array([
    [1, 5],
    [3, 15]
    ]
)

# apply to several vectors and show they all land on the same line
for vec in test_vectors:
    result = np.matmul(dependent_matrix, vec)
    print(f"Squished: {vec} → {result}")

# all results are scalar multiples of each other, all on the same line


# Connection to AI/ML:

# Weight matrices in neural networks are linear transformations.
# Just like our matrix above, each weight matrix takes an input vector
# and transforms it into a new vector in a different space.
# The columns of the weight matrix tell you where each basis direction lands.

# Each layer in a neural network applies one of these transformations.
# Stacking layers means applying multiple transformations one after another,
# each one rotating, scaling, and stretching the data into a new space
# where patterns become easier to separate and classify.

# When you do x @ W in PyTorch that is exactly what we did with np.matmul().
# x is your input vector or batch of vectors.
# W is the weight matrix, the transformation.
# The result is the transformed vector that flows into the next layer.

# This is why matrix multiplication is the core operation in every transformer.
# The attention mechanism is Q @ K.T, a matrix multiplication.
# The feedforward layer is x @ W1 then x @ W2, two matrix multiplications.
# The embedding lookup is a matrix multiplication.
# Literally everything in a neural network is this operation applied repeatedly
# in higher and higher dimensions.