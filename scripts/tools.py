"""
scripts.tools
----------------------------
This module contains the necessary functions to perform experiments with
associative memories.
"""

__author__ = "Arturo Gamino-Carranza"
__version__ = "1.0.0"
__email__ = "arturogamino@hotmail.com"

import os
import copy
import shutil
import glob
import random
import xlsxwriter
import numpy as np
from PIL import Image, ImageOps
from scripts.parameters import paths_directories, associative_memories


def init_directories():
    """Delete and create simulation work directories.

    Args:
        None

    Returns:
        None.
    """

    # Delete work directories
    try:
        shutil.rmtree(os.path.join(os.getcwd(), "simulation"))
    except OSError as error:
        print(f'Error trying to delete work directories:{error.strerror}')

    # Create work directories
    try:
        for name in paths_directories:
            if name not in ["training_input_pattern", "training_output_pattern"]:
                os.makedirs(os.path.join(os.getcwd(), paths_directories[name]))
    except OSError as error:
        print(f'Error trying to create work directories:{error.strerror}')

def training_list(type):
    """Returns a list of images. The images from the source path are saved to
    the target path.

    Args:
        type::str(= "input" or "output")
            Type of training patterns. If type = "X", then the images will be
            saved as X0, X1, X2, ... otherwise the images will be saved as 
            Y0, Y1, Y2, ...

    Returns:
        list_images::list of string
            List of image paths to be used during simulation.
    """

    list_images = []
    count = 0

    source_path = paths_directories["training_input_pattern"]
    target_path = paths_directories["input_pattern"]
    label_name = "X"
    if type == "output":
        source_path = paths_directories["training_output_pattern"]
        target_path = paths_directories["output_pattern"]
        label_name = "Y"

    files = glob.glob(source_path + "*.bmp")

    for filename in sorted(files):
        name_ext = os.path.splitext(filename)
        path_file = target_path + label_name + str(count) + name_ext[1]
        shutil.copy(filename, path_file)
        list_images.append(path_file)
        count += 1

    return list_images

def image_noise(filename, noise_type, noisy_bit):
    """Generates an image with additive, subtractive or mixed noise.

    Args:
        filename::str
            File path of the image to be distorted with noise.
        type_noise::str(= "additive", "subtractive" or "mixed")
            Type of noise used to distort the image.
        noisy_bit::int
            Number of image pixels that will be distorted with noise.

    Returns:
        image::image
            Distorted image with noise.
    """

    image = Image.open(filename)
    matrix = np.array(image, dtype = np.bool_)
    if noise_type == "additive":
        points = np.argwhere(matrix == 0)
    if noise_type == "subtractive":
        points = np.argwhere(matrix == 1)
    if noise_type == "mixed":
        points = np.argwhere(matrix <= 1)
        matrix_aux = np.copy(matrix)

    count = 1
    while count <= noisy_bit:
        index = random.randrange(points.shape[0])
        var_x = points[index][0]
        var_y = points[index][1]

        if noise_type == "additive":
            matrix[var_x, var_y] = 1
        if noise_type == "subtractive":
            matrix[var_x, var_y] = 0
        if noise_type == "mixed":
            matrix[var_x, var_y] = ~matrix_aux[var_x, var_y] & 1

        points = np.delete( points, index, axis=0 )
        count += 1

    matrix = 255 * np.uint8(matrix)
    image = Image.fromarray(matrix)

    return image

def list_images_noise(list_input, noise_type, noisy_bit):
    """Generates a list of noisy images.

    Args:
        list_input::list of str
            List of the paths where the images to be distorted with noise are
            saved.
        noise_type::str(= "additive", "subtractive" or "mixed")
            Type of noise used to distort the image.
        noisy_bit::int
            Number of image pixels that will be distorted with noise.

    Returns:
        list_images::list of string
            List of noisy image paths to be used during simulation.
    """

    list_images = []

    for filename in list_input:
        image = image_noise(filename, noise_type, noisy_bit)
        name_extension = os.path.splitext(os.path.basename(filename))

        path = paths_directories["noise_pattern"] + name_extension[0] + "_" \
                + noise_type + "_" + str(noisy_bit) + name_extension[1]

        image.save(path)
        list_images.append(path)
        list_images = list(list_images)

    return list_images

def list_to_matrix(list_images):
    """Calculates a matrix of column vectors from a list of image paths. Each
    vector corresponds to each image in the list.


    Args:
        list_images::list of str
            List of the paths of the reference images used to calculate
            matrix of column vectors.

    Returns:
        matrix::array object
            Matrix of column vectors corresponding to the images.
    """
    matrix = []

    for filename in list_images:
        image = Image.open(filename)
        vector = np.array(image, dtype=np.uint8).ravel() / 255
        matrix.append(vector)
        image.close()

    matrix = np.array(matrix, dtype=np.uint8)
    matrix = matrix.transpose()

    return matrix

def matrix_to_list(matrix, label_name, size, noise_type, noisy_bit):
    """Generates a list of images from a matrix of column vectors. Each image
    corresponds to each column vector of the matrix. The images are stored in
    the directory path "simulation/recall/".

    Args:
        matrix::array object
            Matrix of column vectors used to generate the list of image paths.
        label_name::str
            Label used to generate the name of the image, which corresponds to
            the name of the associative memory used in the calculation of the
            column vector matrix.
        size::tuple(2)(rows,columns)
            Size of the images to be saved (rows,columns).
            rows::int
                Pixel value of the number of image rows (Height).
            columns::int
                Pixel value of the number of image columns (Width).
        type_noise::str(= "additive", "subtractive" or "mixed")
            Type of noise used to distort the image.
        noisy_bit::int
            Number of image pixels used to distort the image with noise.

    Returns:
        list_images::list of str
            List of the paths of the images stored from matrix of column vectors.
    """

    paths_directories
    array = copy.deepcopy(matrix)
    #rows = dimension[0]
    #cols = dimension[1]
    list_images = []
    number_images = array.shape[1]

    for index in range(number_images):
        #image = 255 * array[:, index].reshape((rows,cols))
        image = 255 * array[:, index].reshape(size)
        image = Image.fromarray(np.uint8(image))

        path = os.path.join(paths_directories["recall_pattern"], "X" + str(index) \
                + "_" + noise_type + "_" + str(noisy_bit) + "_pixels_" \
                + label_name + ".bmp")

        image.save(path)
        list_images.append(path)

    return list_images

def init_name_memories(noise_type):
    """Calculates the associative memories to simulate.

    Args:
        noise_type::str(= "additive", "subtractive" or "mixed")
            Type of noise used to distort the image.

    Returns:
        list_memories::int
            List of memories used in simulation.
    """

    list_memories = []
    for name in associative_memories:
        for attribute in associative_memories[name]["noise"]:
            if attribute == noise_type:
                list_memories.append(name)
    return list_memories

def similarity_list(list_x, list_y):
    """Returns a list of gamma binary similarity distance between the two
    lists of images.

    Citations:
    Mustafa, A. A. (2018). Probabilistic binary similarity  distance for quick
    binary image matching, IET Image Processing, 12(10): 1844-1856.
    DOI: 10.1049/iet-ipr.2017.1333.

    Args:
        list_x::list of str
            List of the paths of the reference images used to calculate
            similarity.
        list_y::list of str
            List of the paths of the images used for comparison with reference
            images.

    Returns:
        similarity::list of float
            List of the floating values of the similarity calculation between
            the images of the two input lists.
    """

    similarity = []

    for (filename_x,filename_y) in zip(list_x ,list_y):
        image_x = Image.open(filename_x)
        image_y = Image.open(filename_y)

        image_x = np.array(image_x, dtype=np.uint8) / 255
        image_y = np.array(image_y, dtype=np.uint8) / 255

        dividend = np.sum(np.absolute(np.subtract(image_x, image_y)))
        divisor = image_x.shape[0] * image_x.shape[0]
        result = abs(1 - (2 * dividend / divisor))
        similarity.append(float(result))

    return similarity

def convert_image_to_grid(scale, invert = False):
    """Converts images of a path to scaled grid images.

    Args:
        scale::int
            Resize 1 pixel by n pixels, where n is the value of the scale
            variable.
        invert::bool(=False)
            Determines whether the image complement is processed. If true, then
            the image complement is obtained.

    Returns:
        None
    """

    for path in paths_directories:
        if path not in ["training_input_pattern", "training_output_pattern"]:
            for filename in glob.glob(paths_directories[path] + "*.bmp"):
                image = Image.open(filename)
                cols, rows = image.size
                image = ImageOps.fit(image, (scale * cols, scale * rows), method = 0)
                matrix = np.array(image, np.uint8)

                if invert:
                    matrix = 255 - matrix

                for index in range(0, cols + 1):
                    matrix = np.insert(matrix, index * scale + index, 0, axis=1)

                for index in range(0, rows + 1):
                    matrix = np.insert(matrix, index * scale + index, [0], axis=0)

                image = Image.fromarray(matrix.astype(np.uint8))
                image.save(filename)

def create_excel(name_memory, number_of_pattern, noise_type, recall_result, perfect_recall_pattern):
    """Generates excel file with simulations result.

    Args:
        name_memory::list of str
            List of name of associative memories used in simulation.
        number_of_patterm::int
            Number of pattern stored in associative memory.
        noise_type::str(= "additive", "subtractive" or "mixed")
            Type of noise used to distort the image.
        recall_result::array object
            Array object where the recall result by associativa memories
            are stored.
        perfect_recall_pattern::array object
            Array object where the number of prefect recall by associativa
            memories are stored.

    Returns:
        None
    """

    # Number of associative memories used to simulation
    number_of_memory = len(name_memory)

    # Generates Excel
    workbook = xlsxwriter.Workbook("Results " + noise_type + ".xlsx")
    worksheet1 = workbook.add_worksheet("Recall " + noise_type)
    worksheet2 = workbook.add_worksheet("Rate " + noise_type)
    bold = workbook.add_format({"bold": 1})

    # Headers
    worksheet1.write("A1", "Memory", bold)
    worksheet1.write("B1", "Noise", bold)
    worksheet1.write("C1", "Noisy pixels", bold)

    worksheet2.write("A1", "Memory", bold)
    worksheet2.write("B1", "Noise", bold)
    worksheet2.write("C1", "Noisy pixeles", bold)
    worksheet2.write("D1", "Perfect recall patterns", bold)
    worksheet2.write("E1", "% rate perfect recall", bold)

    for k in range(number_of_pattern):
        worksheet1.write(0, 3 + k, "Y" + str(k), bold)

    for indexmemory, valmemory in enumerate(name_memory):
        for r in range(2):
            worksheet1.write(1 + 2 * indexmemory + r, 0, associative_memories[valmemory]["name"])
            worksheet1.write(1 + 2 * indexmemory + r, 1, noise_type)
            worksheet1.write(1 + 2 * indexmemory + r, 2, r)

            worksheet2.write(1 + 2 * indexmemory + r, 0, associative_memories[valmemory]["name"])
            worksheet2.write(1 + 2 * indexmemory + r, 1, noise_type)
            worksheet2.write(1 + 2 * indexmemory + r, 2, r)

            worksheet2.write(1 + 2 * indexmemory + r, 3, perfect_recall_pattern[indexmemory][r])
            worksheet2.write(1 + 2 * indexmemory + r, 4, (perfect_recall_pattern[indexmemory][r]) / number_of_pattern * 100)

            for k in range(number_of_pattern):
                worksheet1.write(1 + 2 * indexmemory + r, 3 + k, recall_result[indexmemory][r][k])

    workbook.close()




















