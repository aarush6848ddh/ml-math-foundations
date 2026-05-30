import numpy as np


# Part 1: Find eigenvalues and eigenvectors

v = np.array([
    [1, 3],
    [5, 7]
])

# calculate eigenvectors/eigenvalues
# val contains the eigenvalues: the scalar by which each eigenvector gets scaled
# vec contains the eigenvectors as columns — vec[:, 0] is the first eigenvector
val, vec = np.linalg.eig(v)

print(f"Eigenvalues of v: {val}")
print(f"Eigenvectors of v: {vec}")


# an eigenvector is a special vector that does not get rotated by the transformation
# it only gets stretched or squished along its own span
# the eigenvalue tells you by how much it gets stretched or squished

# a negative eigenvalue means the eigenvector gets flipped in direction
# a large eigenvalue means the transformation strongly amplifies that direction
# an eigenvalue of 1 means the eigenvector stays completely unchanged

# Part 2: Verify the eigenvector property

# for each eigenvector verify that A @ eigenvector = eigenvalue * eigenvector
# this proves the eigenvector only gets scaled, never rotated

for i in range(len(val)):
    print()
    lhs = v @ vec[:, i]       # matrix times eigenvector
    rhs = val[i] * vec[:, i]  # eigenvalue * eigenvector
    print(f"A @ eigenvector {i}: {lhs}")
    print(f"lambda * eigenvector {i}: {rhs}")
    print(f"Are they equal? {np.allclose(lhs, rhs)}")
    print()


# Part 3: Diagonal matrix

# a diagonal matrix has eigenvalues equal to its diagonal entries
# all basis vectors are eigenvectors of a diagonal matrix
diagonal = np.array([
    [3, 0],
    [0, 5]
])

val_d, vec_d = np.linalg.eig(diagonal)
print(f"Eigenvalues of diagonal matrix: {val_d}")
# should be exactly [3, 5], the diagonal entries

# computing the 100th power of a diagonal matrix is trivial
# you just raise each diagonal entry to the 100th power
power_100 = np.linalg.matrix_power(diagonal, 100)
print(f"diagonal^100 top-left entry: {power_100[0][0]}")
print(f"3^100 = {3**100}")
print(f"Are they equal? {power_100[0][0] == 3**100}")

# compare this to how hard it would be to compute the 100th power
# of a non-diagonal matrix, numpy has to do repeated matrix multiplication
non_diagonal = np.array([[1, 3], [5, 7]])
power_100_nd = np.linalg.matrix_power(non_diagonal, 100)
print(f"Non-diagonal^100: {power_100_nd}")
# much harder to compute and impossible to do in your head


# Part 4: Rotation has no real eigenvectors

rotation = np.array([
    [0, -1],
    [1,  0]
])

val_r, vec_r = np.linalg.eig(rotation)
print(f"Eigenvalues of rotation matrix: {val_r}")
# eigenvalues will be complex numbers like 0+1j and 0-1j
# no real eigenvectors exist because every vector gets rotated off its span
print(f"Are eigenvalues complex? {np.iscomplex(val_r).any()}")

# Connection to AI/ML:

# eigenvectors and eigenvalues show up directly in transformer attention
# the attention matrix Q @ K.T has eigenvectors that represent
# the directions in embedding space the model pays most attention to
# the eigenvalues tell you how strongly the model attends in each direction

# PCA uses eigenvectors to find the most important directions in data
# researchers use PCA to compress 512D word embeddings down to 2D
# so they can visualize what the model actually learned about language

# diagonal matrices show up in transformers as scaling operations
# layer normalization scales each dimension independently
# which is essentially multiplying by a diagonal matrix

# the eigenbasis concept eexplains why transformers are so powerful
# attention learns to find th natural axes of meaning in language
# and represent everything in terms of those directions