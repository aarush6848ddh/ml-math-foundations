import numpy as np

# Part 1: 2D Cross Product

v = np.array([3, 1])
w = np.array([1, 2])

# build the matrix with v as column 1 and w as column 2
matrix = np.column_stack((v, w))

# the cross product is the determinant
cross = np.linalg.det(matrix)
print(f"2D cross product: {cross}")
print(f"Area of parallelogram: {abs(cross)}")

# if positive, w is counterclockwise from v
# if negative, w is clockwise from v

# Part 2: 3D Cross Product

v = np.array([3, 1, 0])
w = np.array([1, 2, 7])

# compute cross product
cross = np.cross(v, w)

print(f"Resulting Vector: {cross}")

# length of the cross product vector = area of the parallelogram
print(f"Length (area): {np.linalg.norm(cross)}")

# verify it is perpendicular to both v and w
# dot product with either should be zero
print(f"Perpendicular to v: {np.isclose(np.dot(cross, v), 0)}")
print(f"Perpendicular to w: {np.isclose(np.dot(cross, w), 0)}")


# Part 3: Properties

# perpendicular vectors have larger cross product
v_perp1 = np.array([1, 0, 0])
v_perp2 = np.array([0, 1, 0])

# similar direction vectors have smaller cross product
v_sim1 = np.array([1, 0, 0])
v_sim2 = np.array([1, 0.1, 0])

print(f"Perpendicular cross product length: {np.linalg.norm(np.cross(v_perp1, v_perp2))}")
print(f"Similar direction cross product length: {np.linalg.norm(np.cross(v_sim1, v_sim2))}")

# scaling one vector scales the cross product by the same factor
v = np.array([1, 0, 0])
w = np.array([0, 1, 0])

original = np.linalg.norm(np.cross(v, w))
scaled = np.linalg.norm(np.cross(3 * v, w))

print(f"Original cross product length: {original}")
print(f"Scaled by 3 cross product length: {scaled}")
print(f"Is scaled version 3x original? {np.isclose(scaled, 3 * original)}")


# Connection to AI/ML:

# the cross product is less directly used in AI than the dot product
# but the concepts behind it show up in important places

# the idea that the cross product measures how perpendicular two vectors are
# is the same intuition behind why orthogonal weight matrices are useful
# orthogonal matrices preserve distances and angles during transformation
# which helps gradients flow cleanly during training

# the right hand rule and orientation concepts show up in 3D deep learning
# like point cloud processing and 3D object detection in self driving cars
# models like PointNet use these geometric operations directly on 3D data