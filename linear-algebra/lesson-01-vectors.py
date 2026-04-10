import numpy as np

# A word embedding in AI is a vector that represents the meaning of a word. 
# For example, the word "king" might be represented as a vector in a high-dimensional 
# space, where each dimension corresponds to a different aspect of the word's

# Pretend these are simplified embeddings for the words "king", "queen
# Vectors from a Computer Science students view can be seen as an ordered list of numbers
king = np.array([2.1, 0.3, -1.2, 0.8])
queen = np.array([1.9, 0.5, -1.1, 0.9])

# We can perform vector operations to find relationships between words.

# For example, we can add the vectors for "king" and "queen" to find a new vector 
# that represents the concept of royalty.
result = king + queen

# The result can be interpreted as a new vector that captures the combined 
# meaning of "king" and "queen". For example: Royalty.
print(result)

# We can also scale the vector for "king" by multiplying it by a scalar to find a new vector
# that represents a stronger or weaker version of the concept. For example: Stronger Royalty or Weaker Royalty.
scaled_up = 1.5 * king
flipped = -1 * king 

print("scaled_up:", scaled_up)
print("flipped:", flipped)

# We can also find the difference between the vectors for "king" and "queen" to find a new vector
# that represents the relationship between the two words. For example: Gender Difference.
difference = king - queen

# This new vector can be interpreted as capturing the concept of genders.
print("difference:", difference)

# The famous word embedding analogy
# king - man + woman = queen 
# This is how real LLMs like GPT understand relationships between words and concepts.
# The model learns that "gender direction" in vector space is consistent 

man = np.array([1.0, 0.1, -0.5, 0.2])
woman = np.array([0.9, 0.6, -0.4, 0.8])

analogy_result = king - man + woman 
print("king - man + woman =", analogy_result)

