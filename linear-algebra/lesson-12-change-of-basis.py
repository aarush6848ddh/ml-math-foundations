import numpy as np


# Part 1: Change of Basis Matrix

# Jennifer's basis vectors in our coordinate system
b1 = np.array([2, 1])   # points to the right and up a bit
b2 = np.array([-1, 1])  # points left and up

# change of basis matrix — columns are Jennifer's basis vectors
change_of_basis = np.column_stack((b1, b2))

# Jennifer describes a vector with these coordinates in her system
jennifer_coords = np.array([1, 6])

# convert to our coordinate system
our_coords = change_of_basis @ jennifer_coords
print(f"Jennifer's coords: {jennifer_coords}")
print(f"Our coords: {our_coords}")


# Part 2: Going the Other Way

# a vector in our system
our_vector = np.array([3, 7])

# convert to Jennifer's system using the inverse
inverse_cob = np.linalg.inv(change_of_basis)
jennifer_vector = inverse_cob @ our_vector
print(f"Our vector: {our_vector}")
print(f"Jennifer's vector: {jennifer_vector}")


# Part 3: Translating a Transformation

# 90 degree counterclockwise rotation in our system
rotation = np.array([
    [-1, 0],
    [0, 1]
])

# express the rotation in Jennifer's coordinate system
# formula: inv(change_of_basis) @ rotation @ change_of_basis
jennifer_rotation = np.linalg.inv(change_of_basis) @ rotation @ change_of_basis
print(f"Rotation in our system: {rotation}")
print(f"Rotation in Jennifer's system: {jennifer_rotation}")


# Connection to AI/ML:

# change of basis is exactly what happens inside transformer attention
# when attention computes Q, K, and V it is projecting the input
# into three different coordinate systems simultaneously
# each head in multi-head attention uses a different change of basis matrix
# allowing the model to look at the same information from multiple perspectives

# the formula inv(change_of_basis) @ transformation @ change_of_basis
# shows up directly in how transformers apply attention in different subspaces
# the model learns these basis changes during training
# finding the coordinate systems where patterns in language are easiest to see

# this is why multi-head attention is so powerful
# eight or sixteen heads each seeing the same sentence from a different angle
# is eight or sixteen different change of basis operations happening in parallel