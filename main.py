"""
Implementation of Binary Associative Memories with Complemented Operations in python
@author: Arturo Gamino-Carranza, Directorate of Teaching and Educational Innovation
National Technological Institute of Mexico


Abstract: In this paper, a class of binary associative memory derived from
lattice memories is presented, which is based on the definition of new
complemented binary operations and threshold unary operation. The new learning
method generates the memories M and W, the first is robust to additive noise
and the second is robust to subtractive noise. In the recall step, the memories
converge in a single step and use the same operation as the learning method.
The storage capacity is unlimited and in autoassociative mode there is perfect
recall for the training set. Simulation results suggest that the proposed
memories have better performance compared to other models.
"""

# Imports
import os
import copy
import numpy as np
from PIL import Image
import scripts.tools as tl
import scripts.memories as mm


# Set the type of noise (additive, subtractive, mixed)
noise_type = "additive"

# Set the number of image pixels to be distorted with noise
noisy_pixel = 7
# Set the number of trials to be used
trials = 5

# Set the number of pixels to be scaled at each point in the image
grid = 10

# Delete and create simulation directories
tl.init_directories()

# Generates input and output training pattern list
list_input_pattern = list(tl.training_list("input"))
list_output_pattern = list(tl.training_list("output"))

# Calculates input and output image sizes
image = Image.open(list_input_pattern[0])
size_input_pattern = image.size
image = Image.open(list_output_pattern[0])
size_output_pattern = image.size
image.close()

# Generates matrices of column vectors of input and output training pattern
matrix_input_pattern = tl.list_to_matrix(list_input_pattern)
matrix_output_pattern = tl.list_to_matrix(list_output_pattern)

# Name of associative memories used in simulation according to noise type
memories_to_simulate = tl.init_name_memories(noise_type)

number_of_memory = len(memories_to_simulate)
number_of_pattern = matrix_input_pattern.shape[1]

# Array object to save recall result of associativa memories
matrix_recall_result = np.zeros((number_of_memory, 2, number_of_pattern))

# Array object to save number of perfect recall patterns
matrix_perfect_recall_pattern = np.zeros((number_of_memory, 2))

for index_memory, value_memory in enumerate(memories_to_simulate):
    associative_memory = mm.AssociativeMemory(value_memory, matrix_output_pattern, matrix_input_pattern, size_input_pattern)
    associative_memory.learn()

    for t in range(trials):
        for r in range(2):
            matrix_test = copy.deepcopy(matrix_input_pattern)
            value_noise = 0
            if r > 0:
                # Generates list of noisy image paths for each trial
                list_noise_pattern = list(tl.list_images_noise(list_input_pattern, noise_type, noisy_pixel))
                matrix_test = tl.list_to_matrix(list_noise_pattern)
                value_noise = noisy_pixel

            matrix_recall = associative_memory.recall(matrix_test)

            list_recall =  list(tl.matrix_to_list(matrix_recall, value_memory, size_output_pattern, noise_type, value_noise,"Y"))

            similarity = tl.similarity_list(list_output_pattern, list_recall)
            matrix_recall_result[index_memory][r] += similarity
            matrix_perfect_recall_pattern[index_memory][r] += np.count_nonzero(np.array(similarity) == 1)

matrix_recall_result /= trials
matrix_perfect_recall_pattern /= trials

tl.create_excel(memories_to_simulate, number_of_pattern, noise_type, noisy_pixel, matrix_recall_result, matrix_perfect_recall_pattern)
tl.convert_image_to_grid(grid, invert = True)