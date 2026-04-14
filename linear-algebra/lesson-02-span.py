import numpy as np

# Part 1: Linear Combinations

# define two linear independent basis vectors 
v = np.array([2, 3])
w = np.array([1, 4])

# a*v and b*w = some vector in the plane
scalar_pairs = [(1,1), (2,-1), (0,3), (-1,2), (3,3)]
for a, b in scalar_pairs:
    result = (a * v) + (b * w)
    print(f"a = {a}, b = {b} -> {result}")


# Part 2: Span Demonstration

# two linear independent vectors can reach any point (using vectors from part 1)
matrix = np.column_stack((v, w)) # takes one argument as a tuple of arrays to be stacked as columns
target = np.array([7, 4])

# find the scalars needed such that a*v + b*w = target 
scalars = np.linalg.solve(matrix, target) # finds the coefficients needed to express the target vector as a linear combination of the columns in matrix
print(f"Scalars to reach {target}: {scalars}")

# two linear DEPENDENT vectors can only span a 2d space in a straight line
# one is just a scaled version of another
v_dep = np.array([1,3])
w_dep = np.array([3,9])

# using the same scalars from part 1 to show they will both produce the same 
# direction no matter what 

for a, b in scalar_pairs:
    result = (a * v_dep) + (b * w_dep)
    print(f"Dependent combo: a = {a}, b = {b} -> {result}")
    """
    Output:
    Dependent combo: a = 1, b = 1 -> [ 4 12]
    Dependent combo: a = 2, b = -1 -> [-1 -3]
    Dependent combo: a = 0, b = 3 -> [ 9 27]
    Dependent combo: a = -1, b = 2 -> [ 5 15]
    Dependent combo: a = 3, b = 3 -> [12 36]
    (notice how in all the combos the second number is always 3 times the first)
    (this means that w_dep = 3 * v_dep)
    (this means you are always stuck on the same line through the origin)
    (this is different from part 1 where the independent vectors produced 
    results pointing in all different directions)
    """


# Part 3: Check Linear Dependence

def are_linearly_dependent(v, w) -> bool:
    if np.cross(v, w) == 0: # finds the cross product between two vectors
        return True 
    else:
        return False 

print(are_linearly_dependent(v, w))  # should print False because (2 * 4) - (1 * 3) = 5
print(are_linearly_dependent(v_dep, w_dep))  # should print True because (1 * 9) - (3 * 3) = 0


# Part 4: Standard Basis Vectors

# i_hat and j_hat are the building blocks of all 2D space
i_hat = np.array([1, 0])
j_hat = np.array([0, 1])

# any 2D point is just a linear combination of i_hat and j_hat
target_point = np.array([5, -3])

# with standard basis the scalars are just trivial
# to reach [5, -3] you just scale i_hat by 5 and j_hat by -3
matrix2 = np.column_stack((i_hat, j_hat))
result = np.linalg.solve(matrix2, target_point)
print(f"Reaching {target_point} using standard basis: {result}")

# Connection to AI/ML:
# in transformers each embedding dimension is like a basis vector
# the model learns its own basis for meaning space
# just like we can choose different basis vectors (not just i_hat and j_hat)
# the model learns the most useful directions for representing language





