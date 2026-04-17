import numpy as np


# Part 1: Compute determinants

strech_matrix = np.array([
    [1, -1],
    [0, 1]
])

sqiush_matrix = np.array([
    [0.5, 0],
    [0, 0.5]
])

no_det_matrix = np.array([
    [1, 2],
    [2, 4]
])

# compute determinants
d1 = np.linalg.det(strech_matrix)
d2 = np.linalg.det(sqiush_matrix)
d3 = np.linalg.det(no_det_matrix)

print(f"Stretch matrix determinant: {d1}")
print(f"Squish matrix determinant: {d2}")
print(f"Zero determinant matrix: {d3}")

# Part 2: Negative determinant

neg_d_matrix = np.array([
    [-1, 0],
    [0, 1]
])

det = np.linalg.det(neg_d_matrix)

print(f"Determinant of flipped orientation matrix: {det}")

# a negative determinant means the transformation flipped the orientation of space
# as i-hat and j-hat slowly rotate toward each other, the area between them shrinks
# when they line up completely, the determinant hits 0 (as seen before in part 1)
# if the transformation keeps going past that point, the orientation flips
# and the determinant continues into negative numbers
# the absolute value of the determinant still tells you how much area was scaled
# the negative sign just tells you orientation was flipped

# Part 3: Matrix multiplication property (det(m1 * m2) = det(m1) * det(m2))

m1 = np.array([
    [2, 3],
    [7, 5]
])

m2 = np.array([
    [5, 4],
    [9, 0]
])

# det(m1 * m2)
det1 = np.linalg.det(np.matmul(m1, m2))

# det(m1) * det(m2)
det2 = np.linalg.det(m1) * np.linalg.det(m2)

print(f"Are the equal?: {np.isclose(det1, det2)}")

# Connection to AI/ML:

# A zero determinant in a weight matrix means rank deficiency
# the layer is squishing data into a lower dimension and losing information
# this is why weight initialization matters, you want full rank matrices

# det(M1 * M2) = det(M1) * det(M2) explains the vanishing gradient problem
# if each layer slightly squishes space, multiplying many layers together
# causes the overall transformation to collapse toward zero
