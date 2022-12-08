"""
scripts.memories
----------------------------
This module contains learning and recall algorithms for associative memories.
"""

__author__ = "Arturo Gamino-Carranza"
__version__ = "1.0.0"
__email__ = "arturogamino@hotmail.com"

import os
import copy
import random
import numpy as np
from PIL import Image
from scripts.parameters import paths_directories, associative_memories, offset
from scripts.tools import list_to_matrix

def learn_morphological(pattern_y, pattern_x, type_memory):
    """Returns the computation of the morphological associative memory
    learning matrix.

    Citations:
    Ritter, G. X., Sussner, P. and Díaz de León, J. L. (1998). Morphological
    associative memories, IEEE Transactions on Neural Networks 2(9): 281-293.
    DOI: 10.1109/72.661123.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "mam_max" or "mam_min")
            Type of morphological associative memory used for learning.

    Returns:
        matrix::array object
            Morphological associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.int8)
    matrix_y = copy.deepcopy(pattern_y).astype(np.int8)
    matrix_x = matrix_x.transpose()

    difference = np.subtract(matrix_y[:, :, None], matrix_x)
    if type_memory == "mam_max":
        matrix = np.max(difference, axis = 1)
    if type_memory == "mam_min":
        matrix = np.min(difference, axis = 1)

    return matrix

def recovery_morphological(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the morphological
    associative memory.

    Citations:
    Ritter, G. X., Sussner, P. and Díaz de León, J. L. (1998). Morphological
    associative memories, IEEE Transactions on Neural Networks 2(9): 281-293.
    DOI: 10.1109/72.661123.

    Args:
        memory::array object
            Morphological associative memory learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "mam_max" or "mam_min")
            Type of morphological associative memory used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the morphological associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.int8)
    matrix_memory = copy.deepcopy(memory).astype(np.int8)

    sum = np.add(matrix_memory[:, :, None], matrix_x)
    if type_memory == "mam_max":
        matrix = np.min(sum, axis = 1)
    if type_memory == "mam_min":
        matrix = np.max(sum, axis = 1)

    return matrix

def learn_morphological_plus(pattern_y, pattern_x, type_memory):
    """Returns the computation of the new method of morphological associative
    memory learning matrix.

    Citations:
    Feng, N., Cao, X., Li, S., Ao, L. and Wang, S. (2009). A new method of
    morphological associative memories, Proceedings Emerging Intelligent
    Computing Technology and Applications. With Aspects of Artificial
    Intelligence. "ICIC" 2009, Vol. 5755, Ulsan, South Korea, pp. 407-416.
    DOI: 10.1007/978-3-642-04020-7_43.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "mam_plus_max" or "mam_plus_min")
            Type of the new method of morphological associative memory used
            for learning.

    Returns:
        matrix::array object
            New method of morphological associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.int8)
    matrix_y = copy.deepcopy(pattern_y).astype(np.int8)
    matrix_x = matrix_x.transpose()

    sum = np.add(matrix_y[:, :, None], matrix_x)
    if type_memory == "mam_plus_max":
        matrix = np.max(sum, axis = 1)
    if type_memory == "mam_plus_min":
        matrix = np.min(sum, axis = 1)

    return matrix

def recovery_morphological_plus(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the new method of
    morphological associative memory.

    Citations:
    Feng, N., Cao, X., Li, S., Ao, L. and Wang, S. (2009). A new method of
    morphological associative memories, Proceedings Emerging Intelligent
    Computing Technology and Applications. With Aspects of Artificial
    Intelligence. "ICIC" 2009, Vol. 5755, Ulsan, South Korea, pp. 407-416.
    DOI: 10.1007/978-3-642-04020-7_43.

    Args:
        memory::array object
            New method of morphological associative memory learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "mam_plus_max" or "mam_plus_min")
            Type of the new method of morphological associative memory used
            for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the new method of morphological associative
            memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.int8)
    matrix_memory = copy.deepcopy(memory).astype(np.int8)

    difference = np.subtract(matrix_memory[:, :, None], matrix_x)
    if type_memory == "mam_plus_max":
        matrix = np.min(difference, axis = 1)
    if type_memory == "mam_plus_min":
        matrix = np.max(difference, axis = 1)

    return matrix

def learn_morphological_fuzzy(pattern_y, pattern_x, type_memory):
    """Returns the computation of the new fuzzy morphological associative
    memory learning matrix.

    Citations:
    Wang, S. and Lu, H. (2004). On new fuzzy morphological associative
    memories, IEEE Transactions on Fuzzy Systems 12(3): 316-323.
    DOI: 10.1109/TFUZZ.2004.825977.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "mam_fuzzy_max" or "mam_fuzzy_min")
            Type of new fuzzy mmorphological associative memory used for
            learning.

    Returns:
        matrix::array object
            New fuzzy morphological associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)
    matrix_x = matrix_x.transpose()
    matrix_y += offset
    matrix_x += offset

    quotient = np.divide(matrix_y[:, :, None], matrix_x)
    if type_memory == "mam_fuzzy_max":
        matrix = np.max(quotient, axis = 1)
    if type_memory == "mam_fuzzy_min":
        matrix = np.min(quotient, axis = 1)

    return matrix

def recovery_morphological_fuzzy(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the new fuzzy
    morphological associative memory.

    Citations:
    Wang, S. and Lu, H. (2004). On new fuzzy morphological associative
    memories, IEEE Transactions on Fuzzy Systems 12(3): 316-323.
    DOI: 10.1109/TFUZZ.2004.825977.

    Args:
        memory::array object
            New fuzzy morphological associative memory learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "mam_fuzzy_max" or "mam_fuzzy_min")
            Type of new fuzzy morphological associative memory used for
            recalling.

    Returns:
        matrix::array object
            Matrix recalled by the new fuzzy morphological associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)
    matrix_x += offset

    product = np.multiply(matrix_memory[:, :, None], matrix_x)
    if type_memory == "mam_fuzzy_max":
        matrix = np.min(product, axis = 1)
    if type_memory == "mam_fuzzy_min":
        matrix = np.max(product, axis = 1)

    matrix -= offset

    return matrix

def learn_morphological_reverse_fuzzy(pattern_y, pattern_x, type_memory):
    """Returns the computation of no rounding reverse fuzzy morphological
    associative memory learning matrix.

    Citations:
    Feng, N. and Yao, Y. (2016). No rounding reverse fuzzy morphological
    associative memories, Neural Network World 26(6): 571-587.
    DOI: 10.14311/NNW.2016.26.033.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "mam_rev_fuzzy_max" or "mam_rev_fuzzy_min")
            Type of no rounding reverse fuzzy mmorphological associative memory
            used for learning.

    Returns:
        matrix::array object
            No rounding reverse fuzzy morphological associative memory learning
            matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)
    matrix_x = matrix_x.transpose()
    matrix_y += offset
    matrix_x += offset

    product = np.multiply(matrix_y[:, :, None], matrix_x)
    if type_memory == "mam_rev_fuzzy_max":
        matrix = np.max(product, axis = 1)
    if type_memory == "mam_rev_fuzzy_min":
        matrix = np.min(product, axis = 1)

    return matrix

def recovery_morphological_reverse_fuzzy(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the no rounding
    reverse fuzzy morphological associative memory.

    Citations:
    Feng, N. and Yao, Y. (2016). No rounding reverse fuzzy morphological
    associative memories, Neural Network World 26(6): 571-587.
    DOI: 10.14311/NNW.2016.26.033.

    Args:
        memory::array object
            No rounding reverse fuzzy morphological associative memory
            learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "mam_rev_fuzzy_max" or "mam_rev_fuzzy_min")
            Type of no rounding reverse fuzzy morphological associative memory
            used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the no rounding reverse fuzzy morphological
            associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)
    matrix_x += offset

    quotient = np.divide(matrix_memory[:, :, None], matrix_x)
    if type_memory == "mam_rev_fuzzy_max":
        matrix = np.min(quotient, axis = 1)
    if type_memory == "mam_rev_fuzzy_min":
        matrix = np.max(quotient, axis = 1)

    matrix -= offset

    return matrix

def learn_morphological_logarithmic(pattern_y, pattern_x, type_memory):
    """Returns the computation of the logarithmic and exponential morphological
    associative memory learning matrix.

    Citations:
    Feng, N.-Q., Tian, Y., Wang, X.-F., Song, L.-M., Fan, H.-J. and
    Shuang-Xi, W. (2015). Logarithmic and exponential morphological
    associative memories, Journal of Software 26(7): 1662-1674.
    DOI: 10.13328/j.cnki.jos.004620.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "mam_log_max" or "mam_log_min")
            Type of logarithmic and exponential morphological associative
            memory used for learning.

    Returns:
        matrix::array object
            Logarithmic and exponential morphological associative memory
            learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)
    matrix_x = matrix_x.transpose()
    matrix_y += offset
    matrix_x += offset

    logarithm = np.divide(np.log(matrix_y[:, :, None]), np.log(matrix_x))
    if type_memory == "mam_log_max":
        matrix = np.max(logarithm, axis = 1)
    if type_memory == "mam_log_min":
        matrix = np.min(logarithm, axis = 1)

    return matrix

def recovery_morphological_logarithmic(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the logarithmic and
    exponential morphological associative memory.

    Citations:
    Feng, N.-Q., Tian, Y., Wang, X.-F., Song, L.-M., Fan, H.-J. and
    Shuang-Xi, W. (2015). Logarithmic and exponential morphological
    associative memories, Journal of Software 26(7): 1662-1674.
    DOI: 10.13328/j.cnki.jos.004620.

    Args:
        memory::array object
            Logarithmic and exponential morphological associative memory
            learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "mam_log_max" or "mam_log_min")
            Type of logarithmic and exponential morphological associative
            memory used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the logarithmic and exponential morphological
            associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)
    matrix_x += offset

    exponential = np.power(matrix_x, matrix_memory[:, :, None])
    if type_memory == "mam_log_max":
        matrix = np.min(exponential, axis = 1)
    if type_memory == "mam_log_min":
        matrix = np.max(exponential, axis = 1)

    matrix -= offset

    return matrix

def learn_implicative(pattern_y, pattern_x, type_memory):
    """Returns the computation of the fuzzy implicative associative memory
    learning matrix.

    Citations:
    Sussner, P. and Valle, M. E. (2006). Implicative fuzzy associative
    memories, IEEE Transactions on Fuzzy Systems 14(6): 793-807.
    DOI: 10.1109/TFUZZ.2006.879968.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "ifam_godel_max", "ifam_goguen_max", "ifam_luka_max",
        "ifam_godel_min", "ifam_goguen_min" or "ifam_luka_min")
            Type of fuzzy implicative associative memory used for learning.

    Returns:
        matrix::array object
            Fuzzy implicative associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)
    matrix_x = matrix_x.transpose()

    if type_memory == "ifam_godel_max":
        implicative = np.where(np.greater_equal(matrix_x, matrix_y[:, :, None]), \
                        0, matrix_y[:, :, None])
        matrix = np.max(implicative, axis = 1)
    if type_memory == "ifam_goguen_max":
        implicative = np.where(np.equal(matrix_x, 1), 0, np.maximum(0, \
                        np.divide(np.subtract(matrix_y[:, :, None], matrix_x), \
                        np.subtract(1, matrix_x))))
        matrix = np.max(implicative, axis = 1)
    if type_memory == "ifam_luka_max":
        implicative = np.maximum(0 , np.subtract(matrix_y[:, :, None], matrix_x))
        matrix = np.max(implicative, axis = 1)
    if type_memory == "ifam_godel_min":
        implicative = np.where(np.less_equal(matrix_x, matrix_y[:, :, None]), \
                        1, matrix_y[:, :, None])
        matrix = np.min(implicative, axis = 1)
    if type_memory == "ifam_goguen_min":
        implicative = np.where(np.equal(matrix_x, 0), 1, np.minimum(1, \
                        matrix_y[:, :, None] / matrix_x))
        matrix = np.min(implicative, axis = 1)
    if type_memory == "ifam_luka_min":
        implicative = np.minimum(1, np.add(1 - matrix_x, matrix_y[:, :, None]))
        matrix = np.min(implicative, axis = 1)

    return matrix

def recovery_implicative(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the fuzzy implicative
    associative memory.

    Citations:
    Sussner, P. and Valle, M. E. (2006). Implicative fuzzy associative
    memories, IEEE Transactions on Fuzzy Systems 14(6): 793-807.
    DOI: 10.1109/TFUZZ.2006.879968.

    Args:
        memory::array object
            Fuzzy implicative associative memory learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "ifam_godel_max", "ifam_goguen_max", "ifam_luka_max",
        "ifam_godel_min", "ifam_goguen_min" or "ifam_luka_min")
            Type of fuzzy implicative associative memory used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the fuzzy implicative associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)

    type_max = ["ifam_godel_max", "ifam_goguen_max", "ifam_luka_max"]
    type_min = ["ifam_godel_min", "ifam_goguen_min", "ifam_luka_min"]

    if type_memory in type_max:
        implicative = np.minimum(1, np.add(matrix_memory[:, :, None], matrix_x))
        matrix = np.min(implicative, axis = 1)
    if type_memory in type_min:
        implicative = np.maximum(0, np.add(matrix_memory[:, :, None], matrix_x - 1))
        matrix = np.max(implicative, axis = 1)

    return matrix

def learn_exor(pattern_y, pattern_x, type_memory):
    """Returns the computation of the binary associative memory with complemented
    operations learning matrix.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "mam_max" or "mam_min")
            Type of binary associative memory with complemented operations
            used for learning.

    Returns:
        matrix::array object
            Binary associative memory with complemented operations
            learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.int8)
    matrix_y = copy.deepcopy(pattern_y).astype(np.int8)
    matrix_x = matrix_x.transpose()

    lsb = np.bitwise_xor(matrix_y[:, :, None], matrix_x)
    if type_memory == "exor_max":
        lsb = (~lsb) & 1

    msb = np.bitwise_or((~matrix_y[:, :, None]) & 1, matrix_x)
    if type_memory == "exor_max":
        msb = (~msb) & 1
    msb = np.left_shift(np.uint8(msb), 1)

    matrix = np.max(np.bitwise_or(msb, lsb), axis = 1)
    matrix = np.bitwise_and(matrix, 1)

    return matrix

def recovery_exor(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the binary
    associative memory with complemented operations.

    Args:
        memory::array object
            Binary associative memory with complemented operations
            learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "mam_max" or "mam_min")
            Type of binary associative memory with complemented operations
            used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the binary associative memory with
            complemented operations.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.int8)
    matrix_memory = copy.deepcopy(memory).astype(np.int8)

    lsb = np.bitwise_xor(matrix_memory[:, :, None], matrix_x)
    if type_memory == "exor_max":
        lsb = (~lsb) & 1

    msb = np.bitwise_or((~matrix_memory[:, :, None]) & 1, matrix_x)
    if type_memory == "exor_max":
        msb = (~msb) & 1

    msb = np.left_shift(np.uint8(msb), 1)

    matrix = np.max(np.bitwise_or(msb, lsb), axis = 1)
    matrix = np.bitwise_and(matrix, 1)

    return matrix

def learn_hopfield(pattern_y, pattern_x):
    """Returns the computation of the hopfield associative memory
    learning matrix.

    Citations:
    Hopfield, J. J. (1982). Neural networks and physical systems with emergent
    collective computational abilities, Proceedings of the National Academy of
    Sciences of the United States of America 79(8): 2554-2558.
    DOI: 10.1073/pnas.79.8.2554.
    Xia, G., Tang, Z. and Li, Y. (2004). Hopfield neural network with hysteresis
    for maximum cut problem, Neural Information Processing - Letters and
    Reviews 4(2): 19-26.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.

    Returns:
        matrix::array object
            Hopfield associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)

    # Bipolar matrices
    matrix_x = np.where(np.equal(matrix_x,0), -1, matrix_x)
    matrix_y = np.where(np.equal(matrix_y,0), -1, matrix_y)

    matrix_x = matrix_x.transpose()
    matrix = np.sum(np.multiply(matrix_y[:, :, None], matrix_x), axis = 1)
    np.fill_diagonal(matrix, 0)

    return matrix

def recovery_hopfield(memory, pattern_x, hysterisis = False):
    """Returns the computation of the matrix recalled by the hopfield
    associative memory.

    Citations:
    Hopfield, J. J. (1982). Neural networks and physical systems with emergent
    collective computational abilities, Proceedings of the National Academy of
    Sciences of the United States of America 79(8): 2554-2558.
    DOI: 10.1073/pnas.79.8.2554.
    Xia, G., Tang, Z. and Li, Y. (2004). Hopfield neural network with hysteresis
    for maximum cut problem, Neural Information Processing - Letters and
    Reviews 4(2): 19-26.

    Args:
        memory::array object
            Hopfield associative memory learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        hysterisis::bool(=False)
            Determines if hysteresis is used.

    Returns:
        matrix::array object
            Matrix recalled by the hopfield associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)

    # Bipolar matrix
    matrix_x = np.where(np.equal(matrix_x, 0), -1, matrix_x)

    matrix = np.zeros((matrix_x.shape))

    if hysterisis:
        value = matrix_x.shape[1] - .00001

    number_pattern = matrix_x.shape[1]

    for index in range(number_pattern):
        iteration = 0
        flag = False
        matrix_x_temp = copy.deepcopy(matrix_x[:, index]).reshape((matrix_x.shape[0], 1))

        while True:
            out = np.sum(np.multiply(matrix_memory[:, :, None], matrix_x_temp), axis = 1)

            if hysterisis:
                out = np.where(out < -value, -1, np.where(out > +value, 1, matrix_x_temp))
            else:
                out = np.where(out < 0, -1, np.where(out > 0, 1, matrix_x_temp))

            iteration += 1

            validation = np.alltrue(np.equal(matrix_x_temp, out))

            if (validation or iteration == 2500):
                flag = True
                matrix[:,index] = out.transpose()
                if index == matrix_x.shape[1] - 1:
                    matrix = np.where(np.equal(matrix, -1), 0, matrix)
                    matrix = matrix.astype(np.float32)
                    return matrix
                break

            matrix_x_temp = copy.deepcopy(out)

            if flag:
                break

def learn_gcm(pattern_y, pattern_x):
    """Returns the computation of the globally coupled map associative memory
    learning matrix.

    Citations:
    Ishi, S., Fukumizu, K. and Watanabe, S. (1996). A network of chaotic
    elements for information processing, Neural Networks 9(1): 25-40.
    DOI: 10.1016/0893-6080(95)00100-X.
    Zheng, L. and Tang, X. (2005). A new parameter control method for s-gcm,
    Pattern Recognition Letters 26(7): 939-942. DOI: 10.1016/j.patrec.2004.09.041.
    Wang, T. and Jia, N. (2017). A gcm neural network using cubic logistic map
    for information processing, Neural Computing and Applications 28: 1891-1903.
    DOI: 10.1007/s00521-016-2407-4.
    Wang, T., Jia, N. and Wang, K. (2012). A novel gcm chaotic neural network
    for information processing, Communications in Nonlinear Science and
    Numerical Simulation 17(12): 4846-4855. DOI: 10.1016/j.cnsns.2012.05.011.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.

    Returns:
        matrix::array object
            Globally coupled map associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)

    # Bipolar matrices
    matrix_x = np.where(np.equal(matrix_x,0), -1, matrix_x)
    matrix_y = np.where(np.equal(matrix_y,0), -1, matrix_y)

    matrix_x = matrix_x.transpose()
    matrix = np.sum(np.multiply(matrix_y[:, :, None], matrix_x), axis = 1)
    matrix /= matrix_x.shape[0]

    return matrix

def function_energy(covariance_matrix, pattern_x):
    """Returns the computation of every unit's partial energy function Ei.

    Citations:
    Ishi, S., Fukumizu, K. and Watanabe, S. (1996). A network of chaotic
    elements for information processing, Neural Networks 9(1): 25-40.
    DOI: 10.1016/0893-6080(95)00100-X.
    Zheng, L. and Tang, X. (2005). A new parameter control method for s-gcm,
    Pattern Recognition Letters 26(7): 939-942. DOI: 10.1016/j.patrec.2004.09.041.
    Wang, T. and Jia, N. (2017). A gcm neural network using cubic logistic map
    for information processing, Neural Computing and Applications 28: 1891-1903.
    DOI: 10.1007/s00521-016-2407-4.
    Wang, T., Jia, N. and Wang, K. (2012). A novel gcm chaotic neural network
    for information processing, Communications in Nonlinear Science and
    Numerical Simulation 17(12): 4846-4855. DOI: 10.1016/j.cnsns.2012.05.011.

    Args:
        covariance_matrix::array object
            Covariance matrix of the set of memorized patterns.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.

    Returns:
        energy::array object
            Unit's partial energy.
    """
    covariance = copy.deepcopy(covariance_matrix).astype(np.float32)
    input = copy.deepcopy(pattern_x).astype(np.float32)

    partial = np.sum(np.multiply(covariance[:, :, None], input), axis = 1)
    energy = np.multiply(partial, -input)

    return energy

def recovery_gcm(memory, pattern_x, pattern_y, type_memory):
    """Returns the computation of the matrix recalled by the globally coupled
    map associative memory.

    Citations:
    Ishi, S., Fukumizu, K. and Watanabe, S. (1996). A network of chaotic
    elements for information processing, Neural Networks 9(1): 25-40.
    DOI: 10.1016/0893-6080(95)00100-X.
    Zheng, L. and Tang, X. (2005). A new parameter control method for s-gcm,
    Pattern Recognition Letters 26(7): 939-942. DOI: 10.1016/j.patrec.2004.09.041.
    Wang, T. and Jia, N. (2017). A gcm neural network using cubic logistic map
    for information processing, Neural Computing and Applications 28: 1891-1903.
    DOI: 10.1007/s00521-016-2407-4.
    Wang, T., Jia, N. and Wang, K. (2012). A novel gcm chaotic neural network
    for information processing, Communications in Nonlinear Science and
    Numerical Simulation 17(12): 4846-4855. DOI: 10.1016/j.cnsns.2012.05.011.

    Args:
        memory::array object
            Globally coupled map associative memory learning matrix.
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "s_gcm", "s_gcm_2", "cl_gcm" or "si_gcm")
            Type of globally coupled map associative memory used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the globally coupled map associative memory.
    """

    while True:
        rand_positive = np.random.rand()
        if 0< rand_positive <.1499:
            break

    if type_memory in ["s_gcm", "s_gcm_2"]:
        alpha_min = 3.4
        alpha_max = 4.0
        beta = 2.0
        epsilon = 0.1
        alpha = 3.5
        rand_negative = -np.sqrt( (alpha_min-1)/alpha_min ) - rand_positive
        rand_positive = np.sqrt( (alpha_min-1)/alpha_min ) + rand_positive
        if type_memory == "s_gcm_2":
            value_r = 0.7
    if type_memory == "cl_gcm":
        alpha_min = 1.8
        alpha_max = 3.0
        beta = 0.5
        epsilon = 0.2
        alpha = 2.8
        rand_negative = -1 + rand_positive
        rand_positive = 1 - rand_positive
    if type_memory == "si_gcm":
        alpha_min = 0.8
        alpha_max = 1.8
        beta = 1
        epsilon = 0.1
        alpha = 1.45
        value_r = 0.7
        rand_negative = -0 + rand_positive
        rand_positive = 0 - rand_positive

    matrix_memory = copy.deepcopy(memory).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)

    # Bipolar matrix
    matrix_y = np.where(np.equal(matrix_y, 0), -1, matrix_y)

    matrix = np.zeros((matrix_y.shape))
    number_pattern = matrix_y.shape[1]

    for index in range(number_pattern):
        flag = False
        iteration = 0
        matrix_x = copy.deepcopy(pattern_x[:, index]).reshape((pattern_x.shape[0], 1))

        # function V convert a binary vector {-1,1} into V(I) [-1,1]
        # apply only for S-GCM
        matrix_x = np.where( np.equal(matrix_x, 1), rand_positive, rand_negative)

        if type_memory in ["s_gcm_2", "si_gcm"]:
            energy_before = 0

        count = 0

        while True:
            # if function != 's_gcm' or ( function == 's_gcm' and ite % 16 == 0 ):
            # partial energy function Ei = -x_i * ( E w_ij * x_j )
            energy = function_energy(matrix_memory, matrix_x)

            if type_memory in ["s_gcm", "cl_gcm"]:
                # alpha function a = a_i+(a_i-a_min)*tanh(b*E)
                alpha = alpha + (alpha - alpha_min) * np.tanh(beta * energy)
                alpha = np.minimum( np.maximum(alpha, alpha_min), alpha_max)

            if type_memory in ["s_gcm_2", "si_gcm"]:
                # delta energy = E_i(t) -E_i(t-1)
                delta_energy = energy - energy_before

                # alpha function a = a_i+(a_i-a_min)*tanh(b*E)*[1+r*(1-tanh(b*abs(D_E)))]
                alpha = (alpha + (alpha - alpha_min) * np.tanh(beta * energy) \
                    * (1 + value_r * (1 - np.tanh(beta * abs(delta_energy)))))
                alpha = np.minimum(np.maximum(alpha, alpha_min), alpha_max)

                # E_i(t-1)
                energy_before = copy.deepcopy(energy)

            if type_memory in ["s_gcm", "s_gcm_2"]:
                # cubic function f_i(x) = a_i*x^3 - a_i*x + x
                function = (alpha * (matrix_x ** 3)) - (alpha * matrix_x) + matrix_x
            elif type_memory == "cl_gcm":
                # cubic function f_i(x) = a_i*x(1-x^2)
                function = (alpha*matrix_x) * (1 - (matrix_x ** 2))
            elif type_memory == "si_gcm":
                # sinoidal function f_i(x) = a_i*sin(ai*pi*x)
                function = alpha * np.sin(alpha * np.pi * matrix_x)

            # dynamic function x_i(t+1) = (1-e)*f_i(x_i(t)) + e/N * E f_j(x_j(t))
            function_partial = np.sum(function) / matrix_x.shape[0]
            out_x = (1 - epsilon) * function + epsilon * function_partial

            # binary coding function C = 1 if x_i >= x and C = -1 otherwise
            out = np.where(np.greater_equal(out_x, 0), 1, -1)

            iteration += 1

            if iteration == 2500:
                flag = True
            else:
                if type_memory in ["s_gcm", "s_gcm_2", "si_gcm"]:
                    validation = np.isclose(alpha, alpha_min)
                if type_memory == "cl_gcm":
                    comparation_x = np.where(np.greater_equal(matrix_x, 0), 1, -1)
                    validation = np.isclose(comparation_x, out)

                if np.alltrue(validation):
                    count += 1
                else:
                    count = 0

                if count == 10:
                    flag = True

            matrix_x = copy.deepcopy(out_x)

            if flag:
                matrix[:, index] = out.transpose()
                if index == matrix_y.shape[1]-1:
                    matrix = np.where(np.equal(matrix, -1), 0, matrix)
                    matrix.astype(np.float32)
                    return matrix
                break

def learn_pcsmn(pattern_y, pattern_x):
    """Returns the computation of the parametrically coupled sine map network
    learning matrix.

    Citations:
    Lee, G. and Farhat, N. H. (2001). Parametrically coupled sine map networks,
    International Journal of Bifurcation and Chaos 11(07): 1815-1834.
    DOI: 10.1142/S0218127401003048.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.

    Returns:
        matrix::array object
            Parametrically coupled sine map network learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)

    # Bipolar matrices
    matrix_x = np.where(np.equal(matrix_x,0), -1, matrix_x)
    matrix_y = np.where(np.equal(matrix_y,0), -1, matrix_y)

    matrix_x = matrix_x.transpose()
    matrix = np.sum(np.multiply(matrix_y[:, :, None], matrix_x), axis = 1)

    return matrix

def recovery_pcsmn(memory, pattern_x, pattern_y, type_memory):
    """Returns the computation of the matrix recalled by the parametrically
    coupled sine map network.

    Citations:
    Lee, G. and Farhat, N. H. (2001). Parametrically coupled sine map networks,
    International Journal of Bifurcation and Chaos 11(07): 1815-1834.
    DOI: 10.1142/S0218127401003048.

    Args:
        memory::array object
            Parametrically coupled sine map network learning matrix.
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "pcsmn_1" or "pcsmn_1")
            Type of parametrically coupled sine map network used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the parametrically coupled sine map network.
    """

    # copia las matrices por si hay que modificarlas
    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)

    # Bipolar matrix
    matrix_y = np.where(np.equal(matrix_y, 0), -1, matrix_y)

    if type_memory == "pcsmn_1":
        eta_zero = 1.0+0.03
        alpha = 1
        epsilon = 0.14
    if type_memory == "pcsmn_2":
        eta_zero = 1.0+0.0
        alpha = 0.6
        epsilon = 0.2

    matrix = np.zeros((matrix_x.shape))
    number_pattern = matrix_y.shape[1]

    for index in range(number_pattern):
        iteration = 0
        count = 0
        flag = False
        matrix_x_temp = copy.deepcopy(matrix_x[:, index]).reshape((matrix_x.shape[0], 1))

        if type_memory == 'pcsmn_2':
            function_h_before = 0

        while True:
            # sum of w_ij*x_j
            partial = np.sum(np.multiply(matrix_memory[:, :, None], matrix_x_temp), axis = 1)

            if type_memory == 'pcsmn_1':
                partial = ((alpha / matrix_x_temp.shape[0]) * matrix_x_temp) * partial
                partial = np.minimum(np.maximum(partial, -epsilon), +epsilon)
                eta = eta_zero - partial
                out_x = eta * np.sin(matrix_x_temp * np.pi)

            if type_memory == 'pcsmn_2':
                partial = alpha * partial
                u_i = np.minimum(np.maximum(partial, -epsilon), +epsilon)
                out_x = eta_zero * np.sin(2 * np.pi * matrix_x_temp) - u_i

            out = np.where( np.greater_equal(out_x,0), 1, -1 )

            iteration += 1

            if type_memory == 'pcsmn_2':
                function_h = np.sum( np.multiply(matrix_memory[:, :, None], out), axis=1 )
                function_h = function_h * out
                function_h = -1 * np.sum(function_h)

            if type_memory == "pcsmn_1" and iteration > 900:
                validation = np.alltrue(np.equal(np.sign(matrix_x_temp), np.sign(out_x)))
                if validation:
                    count += 1
                else:
                    count = 0
            if type_memory == "pcsmn_2":
                if function_h == function_h_before:
                    count += 1
                else:
                    count = 0

                function_h_before = copy.deepcopy(function_h)

            matrix_x_temp = copy.deepcopy(out_x)

            if count == 101 or iteration == 2500:
                flag = True
                matrix[:, index] = out.transpose()
                if index == matrix_y.shape[1]-1:
                    matrix = np.where(np.equal(matrix, -1), 0, matrix)
                    matrix = matrix.astype(np.float32)
                    return matrix

            if flag:
                break

def learn_fuzzy(pattern_y, pattern_x, type_memory):
    """Returns the computation of the fuzzy associative memory learning matrix.

    Citations:
    Zhang, S., Lin, S. and Chen, C. (1993). Improved model of optical fuzzy
    associative memory, Optics Letters 18(21): 1837-1839.
    DOI: 10.1364/OL.18.001837.
    Chung, F.-L. and Lee, T. (1994). Towards a high capacity fuzzy associative
    memory model, Proceedings of 1994 “IEEE” International Conference on Neural
    Networks (ICNN’94), Vol. 3, Florida, United States, pp. 1595-1599.
    DOI: 10.1109/ICNN.1994.374394.
    Xiao, P., Yang, F. and Yu, Y. (1997). Max-min encoding learning algorithm
    for fuzzy max-multiplication associative memory networks, Proceedings 1997
    “IEEE” International Conference on Systems, Man, and Cybernetics.
    Computational Cybernetics and Simulation, Vol. 4, Orlando, United States,
    pp. 3674-3679. DOI: 10.1109/ICSMC.1997.633240.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        type_memory::str(= "fam_zhang", "fam_chunglee" or "fam_xiao")
            Type of fuzzy associative memory used for learning.

    Returns:
        matrix::array object
            Fuzzy associative memory learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)
    matrix_x = matrix_x.transpose()

    if type_memory in ["fam_zhang", "fam_chunglee"]:
        matrix = np.min(np.where(np.less_equal(matrix_x[:, :, None], matrix_y), \
                    1, matrix_y), axis = 1)
    if type_memory == "fam_xiao":
        matrix = np.min(np.where(np.equal(matrix_x[:, :, None], matrix_y), 1, \
                    np.divide(np.minimum(matrix_x[:, :, None], matrix_y), \
                    np.maximum(matrix_x[:, :, None], matrix_y))), axis = 1)

    return matrix

def recovery_fuzzy(memory, pattern_x, type_memory):
    """Returns the computation of the matrix recalled by the fuzzy associative
    memory.

    Citations:
    Zhang, S., Lin, S. and Chen, C. (1993). Improved model of optical fuzzy
    associative memory, Optics Letters 18(21): 1837-1839.
    DOI: 10.1364/OL.18.001837.
    Chung, F.-L. and Lee, T. (1994). Towards a high capacity fuzzy associative
    memory model, Proceedings of 1994 “IEEE” International Conference on Neural
    Networks (ICNN’94), Vol. 3, Florida, United States, pp. 1595-1599.
    DOI: 10.1109/ICNN.1994.374394.
    Xiao, P., Yang, F. and Yu, Y. (1997). Max-min encoding learning algorithm
    for fuzzy max-multiplication associative memory networks, Proceedings 1997
    “IEEE” International Conference on Systems, Man, and Cybernetics.
    Computational Cybernetics and Simulation, Vol. 4, Orlando, United States,
    pp. 3674-3679. DOI: 10.1109/ICSMC.1997.633240.

    Args:
        memory::array object
            Fuzzy associative memory learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.
        type_memory::str(= "fam_zhang", "fam_chunglee" or "fam_xiao")
            Type of fuzzy associative memory used for recalling.

    Returns:
        matrix::array object
            Matrix recalled by the fuzzy associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)

    if type_memory == "fam_zhang":
        matrix = np.max(np.minimum(matrix_x[:, :, None], matrix_memory), axis = 1)
    if type_memory == "fam_chunglee":
        matrix = np.max(np.maximum(0, np.add(matrix_x[:, :, None], matrix_memory - 1)), \
                    axis = 1)
    if type_memory == "fam_xiao":
        matrix = np.max(np.multiply(matrix_x[:, :, None], matrix_memory), axis = 1)

    return matrix

def learn_fuzzy_liu(pattern_y, pattern_x):
    """Returns the computation of the max-min fuzzy neural network with
    threshold learning matrix.

    Citations:
    Liu, P. (1999). The fuzzy associative memory of max-min fuzzy neural
    network with threshold, Fuzzy Sets and Systems 107(2): 147-157.
    DOI: 10.1016/S0165-0114(97)00352-7.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.

    Returns:
        matrix::array object
            Max-min fuzzy neural network with threshold learning matrix.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_y = copy.deepcopy(pattern_y).astype(np.float32)

    # G_ij = { p E K | (x_i)^p > (y_j)^p}
    matrix_g = np.greater(matrix_y[:, :, None], matrix_x.transpose())
    matrix = np.ones((matrix_y.shape[0], matrix_x.shape[0]))

    for indices in np.ndindex(matrix_y.shape[0], matrix_x.shape[0]):
        index_i = indices[0]
        index_j = indices[1]

        if np.count_nonzero(matrix_g[index_i, :, index_j]) > 0:
            matrix_temp = matrix_g[index_i, :, index_j] * matrix_x[index_j, :]
            coordinates = np.where(matrix_g[index_i, :, index_j] > 0)
            matrix[index_i, index_j] = np.min(matrix_temp[coordinates])

    return matrix

def recovery_fuzzy_liu(memory, pattern_x):
    """Returns the computation of the matrix recalled by the max-min fuzzy
    neural network with threshold.

    Citations:
    Liu, P. (1999). The fuzzy associative memory of max-min fuzzy neural
    network with threshold, Fuzzy Sets and Systems 107(2): 147-157.
    DOI: 10.1016/S0165-0114(97)00352-7.

    Args:
        memory::array object
            Max-min fuzzy neural network with threshold learning matrix.
        pattern_x::array object
            Matrix of column vectors of the input patterns to be recalled by
            the associative memory.

    Returns:
        matrix::array object
            Matrix recalled by the max-min fuzzy neural network with threshold.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.float32)
    matrix_memory = copy.deepcopy(memory).astype(np.float32)
    matrix_x = matrix_x.transpose()

    matrix = np.max(np.minimum(matrix_x[:, :, None], matrix_memory), axis = 1)

    return matrix

def save_kernel(matrix, type_memory, size):
    """Save the binary kernel for binary associative memories.

        Args:
            matrix::array object
                Matrix of column vectors corresponding to the kernel patterns
            type_memory::str(= "kernel_mam_heuristic" or "kernel_exor")
                Type of associative memory used.
            size::tuple(2)(rows,columns)
                Size of the images to be saved (rows,columns).
                rows::int
                    Pixel value of the number of image rows (Height).
                columns::int
                    Pixel value of the number of image columns (Width).

        Returns:
            None
        """

    for i in range(matrix.shape[1]):
        image = 255 * matrix[:, i].reshape(size)
        image = Image.fromarray(np.uint8(image))

        path = os.path.join(paths_directories["kernel_pattern"], "Z" + str(i) + "_" \
                + type_memory +".bmp")
        image.save(path)

def kernel_hattori(pattern_y, pattern_x, size):
    """Determines the binary kernel for binary associative memories with
    complemented operations, using Hattori's algorithm.

    Citations:
    Hattori, M., Fukui, A. and Ito, H. (2002). A fast method of constructing
    kernel patterns for morphological associative memory, Proceedings.
    9th International Conference on Neural Information Processing, 2002.
    “ICONI” 02, Vol. 2, Singapore, Singapore, pp. 1058-1063.
    DOI: 10.1109/ICONIP.2002.1198222.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        size::tuple(2)(rows,columns)
                Size of the images to be saved (rows,columns).
                rows::int
                    Pixel value of the number of image rows (Height).
                columns::int
                    Pixel value of the number of image columns (Width).

    Returns:
        matrix_z::array object
            Kernel for binary associative memories with complemented operations.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.uint16)
    matrix_y = copy.deepcopy(pattern_y).astype(np.uint16)

    # Hattori's algorithm condition P1
    frecuency = np.count_nonzero(matrix_x, axis = 1)
    frecuency = frecuency.reshape(matrix_x.shape[0], 1)
    matrix_z = matrix_x * frecuency
    matrix_z[matrix_z != 1 ] = 0

    # Kernel test computed
    memory_mzz = AssociativeMemory("exor_max", matrix_z, matrix_z, size)
    memory_mzz.learn()
    memory_wzy = AssociativeMemory( "exor_min", matrix_y, matrix_z, size)
    memory_wzy.learn()
    matrix_mzz = memory_mzz.recall(matrix_x)
    matrix_out = memory_wzy.recall(matrix_mzz)

    validation = np.alltrue(np.isclose(matrix_y, matrix_out))
    if validation:
        save_kernel(matrix_z, "kernel_exor", size)
        return matrix_z

    # Hattori's algorithm condition P2
    number_pattern = matrix_x.shape[1]
    for k in range(number_pattern):
        validation = np.alltrue(np.isclose(matrix_y[:, k], matrix_out[:, k]))
        if not validation:
            listcontain = []
            for p in range(number_pattern):
                if p != k:
                    validation = np.alltrue(matrix_x[:, k] <= matrix_x[:, p])
                    if validation:
                        listcontain.append(p)

            matrix_x_copy = copy.deepcopy(matrix_x)
            for p in listcontain:
                matrix_x_copy[:, p] = 0

            vector_x = matrix_x_copy[:, k] * frecuency.transpose()
            minimo = np.amin(vector_x, where = vector_x > 0, initial = number_pattern + 1)
            varx, coord = np.where(vector_x == minimo)
            pointer = random.randint(0, len(coord) - 1)
            row = coord[pointer]
            matrix_z[row, k] = 1

            temp = np.delete(coord, pointer)
            coord = temp
            coordz = np.where(matrix_x_copy[row, :] > 0)
            vecz = coordz[0]
            vecz = np.setdiff1d(vecz, [k])
            vecz = np.setdiff1d(vecz, listcontain)

            while len(vecz):
                for i in coord:
                    if matrix_z[i, k] == 0:
                        coordi = np.where(matrix_x_copy[i, :] > 0)
                        veci = coordi[0]
                        veci = np.setdiff1d(veci, [k])
                        veci = np.setdiff1d(veci, listcontain)

                        if len(np.setdiff1d(vecz, veci)) > 0:
                            matrix_z[i, k]=1
                            vecz = np.intersect1d(vecz,veci)

                if len(vecz) > 0:
                    while True:
                        minimo += 1
                        if minimo in vector_x:
                            varx, coord = np.where(vector_x == minimo)
                            break

    # Kernel test computed
    memory_mzz = AssociativeMemory("exor_max", matrix_z, matrix_z, size)
    memory_mzz.learn()
    memory_wzy = AssociativeMemory( "exor_min", matrix_y, matrix_z, size)
    memory_wzy.learn()
    matrix_mzz = memory_mzz.recall(matrix_x)
    matrix_out = memory_wzy.recall(matrix_mzz)

    validation = np.alltrue(np.isclose(matrix_y, matrix_out))
    if validation:
        save_kernel(matrix_z, "kernel_exor", size)
        return matrix_z

    iteration=1
    while True:
        # Hattori's algorithm condition P3
        for k in range(number_pattern):
            validation = np.alltrue(np.isclose(matrix_y[:, k], matrix_out[:, k]))
            if not validation:
                matrix_z[:, k] = matrix_mzz[:, k]

        # Kernel test computed
        memory_mzz = AssociativeMemory("exor_max", matrix_z, matrix_z, size)
        memory_mzz.learn()
        memory_wzy = AssociativeMemory( "exor_min", matrix_y, matrix_z, size)
        memory_wzy.learn()
        matrix_mzz = memory_mzz.recall(matrix_x)
        matrix_out = memory_wzy.recall(matrix_mzz)

        validation = np.alltrue(np.isclose(matrix_y, matrix_out))
        if validation:
            save_kernel(matrix_z, "kernel_exor", size)
            return matrix_z

        # Hattori's algorithm condition P3
        coord = []
        for k in range(number_pattern):
            validation = np.alltrue(np.isclose(matrix_y[:, k], matrix_out[:, k]))
            if not validation:
                coord.append(k)

        for k in coord:
            vector_x = matrix_x[:, k] * frecuency.transpose()
            minimo = np.amin(vector_x, where = vector_x > 0, initial = number_pattern + 1)

            flag=True
            for m in range(minimo, number_pattern):
                if m in vector_x:
                    varx, coordz = np.where(vector_x == m)

                    for i in coordz:
                        if np.sum(matrix_z[i, :]) == 0:
                            matrix_z[i, k] = 1
                            flag = False
                            break

                if flag is False:
                    break

            if flag is True:
                for m in range(minimo, number_pattern):
                    if m in vector_x:
                        varx, coordz = np.where(vector_x == m)

                        for i in coordz:
                            if matrix_z[i, k] == 0:
                                matrix_z[i, k] = 1
                                flag = False
                                break

                    if flag is False:
                        break

        # Kernel test computed
        memory_mzz = AssociativeMemory("exor_max", matrix_z, matrix_z, size)
        memory_mzz.learn()
        memory_wzy = AssociativeMemory( "exor_min", matrix_y, matrix_z, size)
        memory_wzy.learn()
        matrix_mzz = memory_mzz.recall(matrix_x)
        matrix_out = memory_wzy.recall(matrix_mzz)

        validation = np.alltrue(np.isclose(matrix_y, matrix_out))
        if validation or iteration == 100:
            save_kernel(matrix_z, "kernel_exor", size)
            return matrix_z

        iteration += 1

def kernel_heuristic(pattern_y, pattern_x, size):
    """Determines the binary kernel for morphological associative memories,
    using the heuristic trial-and-error algorithm.

    Citations:
    Ritter, G. X., Sussner, P. and Díaz de León, J. L. (1998). Morphological
    associative memories, IEEE Transactions on Neural Networks 2(9): 281-293.
    DOI: 10.1109/72.661123.

    Args:
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        size::tuple(2)(rows,columns)
                Size of the images to be saved (rows,columns).
                rows::int
                    Pixel value of the number of image rows (Height).
                columns::int
                    Pixel value of the number of image columns (Width).

    Returns:
        matrix_z::array object
            Kernel for morphological associative memory.
    """

    matrix_x = copy.deepcopy(pattern_x).astype(np.uint16)
    matrix_y = copy.deepcopy(pattern_y).astype(np.uint16)
    matrix_z = np.zeros((matrix_x.shape))
    frecuency = np.count_nonzero(matrix_x, axis = 1)
    frecuency = frecuency.reshape(matrix_x.shape[0], 1)
    number_pattern = matrix_x.shape[1]

    for k in reversed(range(number_pattern)):
        minimo_x = 1
        minimo_z = 0
        flag = True
        count = 0
        listcontain = []

        for p in range(number_pattern):
            if p != k:
                if np.alltrue(matrix_x[:, p] >= matrix_x[:, k]):
                    listcontain.append(p)

        while flag:
            for coordinates in range(matrix_x.shape[0]):
                if matrix_x[coordinates, k] == 1 and matrix_z[coordinates, k] == 0:
                    flag_contain = 0
                    for p in listcontain:
                        if matrix_z[coordinates, p] == 1:
                            flag_contain = 1
                    if np.count_nonzero(matrix_z[coordinates, :]) == minimo_z and flag_contain != 1:
                        if count == 5:
                            flag = False
                            break
                        matrix_z[coordinates, k] = 1
                        count += 1

            minimo_z +=1
            if minimo_z == matrix_x.shape[0]:
                minimo_x += 1
                minimo_z = 0

    umbral = 2
    for k in reversed(range(number_pattern)):
        count = random.randint(0, umbral)
        flag = True

        while flag:
            coordinates = random.randint(0, matrix_x.shape[0]-1)
            if count <= umbral:
                if count == umbral:
                    flag = False
                else:
                    if matrix_x[coordinates, k] == 1:
                        matrix_z[coordinates, k] = 1 - matrix_z[coordinates, k]
                        count += 1

    # Kernel test computed
    memory_mzz = AssociativeMemory("mam_max", matrix_z, matrix_z, size)
    memory_mzz.learn()
    memory_wzy = AssociativeMemory( "mam_min", matrix_y, matrix_z, size)
    memory_wzy.learn()
    matrix_mzz = memory_mzz.recall(matrix_x)
    matrix_out = memory_wzy.recall(matrix_mzz)

    validation = np.alltrue(np.isclose(matrix_y, matrix_out))
    if validation:
        save_kernel(matrix_z, "kernel_mam_heuristic", size)
        return matrix_z

    iteration = 0
    percent = .55
    maxpixel = percent * np.max(np.count_nonzero(matrix_x, axis = 0))

    while True:
        for k in reversed(range(number_pattern)):
            validation = np.alltrue(np.isclose(matrix_y[:, k], matrix_out[:, k]))
            if not validation:
                matrix_z[:, k] = matrix_mzz[:, k]

        for k in reversed(range(number_pattern)):
            if np.count_nonzero(matrix_z[:, k]) > maxpixel:
                flag = True

                while flag:
                    coordinates = random.randint(0, matrix_x.shape[0]-1)
                    if matrix_z[coordinates, k] == 1:
                        matrix_z[coordinates, k] = 0
                        if not np.count_nonzero(matrix_z[:, k]) > maxpixel:
                            flag = False

        # Kernel test computed
        memory_mzz = AssociativeMemory("mam_max", matrix_z, matrix_z, size)
        memory_mzz.learn()
        memory_wzy = AssociativeMemory( "mam_min", matrix_y, matrix_z, size)
        memory_wzy.learn()
        matrix_mzz = memory_mzz.recall(matrix_x)
        matrix_out = memory_wzy.recall(matrix_mzz)

        validation = np.alltrue(np.isclose(matrix_y, matrix_out))
        if validation:
            save_kernel(matrix_z, "kernel_mam_heuristic", size)
            return matrix_z

        coordinates = []
        for k in range(number_pattern):
            validation = np.alltrue(np.isclose(matrix_y[:, k], matrix_out[:, k]))
            if not validation:
                coordinates.append(k)

        for k in coordinates:
            vector_x = matrix_x[:, k] * frecuency.transpose()
            minimo = np.amin(vector_x, where = vector_x > 0, initial = number_pattern + 1)

            flag = True
            for m in range(minimo, number_pattern):
                if m in vector_x:
                    var_x, coord_z = np.where(vector_x == m)

                    for i in coord_z:
                        if np.sum(matrix_z[i, :]) == 0:
                            matrix_z[i, k] = 1
                            flag = False
                            break

                if flag is False:
                    break

            if flag is True:
                for m in range(minimo, number_pattern):
                    if m in vector_x:
                        var_x, coord_z = np.where(vector_x == m)

                        for i in coord_z:
                            if matrix_z[i, k] == 0:
                                matrix_z[i, k] = 1
                                flag = False
                                break

                    if flag is False:
                        break

        for k in reversed(range(number_pattern)):
            validation = np.count_nonzero(matrix_z[:, k]) > maxpixel
            if validation:
                flag = True

                while flag:
                    coordinates = random.randint(0, matrix_x.shape[0] - 1)
                    if matrix_z[coordinates, k] == 1:
                        matrix_z[coordinates, k] = 0
                        validation = np.count_nonzero(matrix_z[:,k]) > maxpixel
                        if not validation:
                            flag = False

        # Kernel test computed
        memory_mzz = AssociativeMemory("mam_max", matrix_z, matrix_z, size)
        memory_mzz.learn()
        memory_wzy = AssociativeMemory( "mam_min", matrix_y, matrix_z, size)
        memory_wzy.learn()
        matrix_mzz = memory_mzz.recall(matrix_x)
        matrix_out = memory_wzy.recall(matrix_mzz)

        validation = np.alltrue(np.isclose(matrix_y, matrix_out))
        if validation:
            save_kernel(matrix_z, "kernel_mam_heuristic", size)
            return matrix_z

        iteration +=1


class AssociativeMemory:
    """A class to represent a associative memory.

    Attributes:
        associative_memory::str(="hopfield", "hopfield_hysterisis", "pcsmn_1",
        "pcsmn_2", "s_gcm", "s_gcm_2", "cl_gcm", "si_gcm", "fam_zhang", "fam_xiao",
        "fam_chunglee", "fam_liu", "mam_max", "mam_plus_max", "mam_fuzzy_max",
        "mam_rev_fuzzy_max", "mam_log_max", "ifam_godel_max", "ifam_goguen_max",
        "ifam_luka_max", "exor_max", "mam_min", "mam_plus_min", "mam_fuzzy_min",
        "mam_rev_fuzzy_min", "mam_log_min", "ifam_godel_min", "ifam_goguen_min",
        "ifam_luka_min", "exor_min", "kernel_exor" or "kernel_mam_heuristic")
            associative memory codename.
        pattern_x::array object
            Matrix of column vectors corresponding to the input patterns
            of the training set.
        pattern_y::array object
            Matrix of column vectors corresponding to the output patterns
            of the training set.
        size_x::tuple(2)(rows,columns)
                Size of the kernel images to be saved (rows,columns).
                rows::int
                    Pixel value of the number of image rows (Height).
                columns::int
                    Pixel value of the number of image columns (Width).

    Methods:
        learn():
            Calculates the associative memory learning matrix.
        recall(pattern_x::array object):
            Calculates the matrix recalled by the associative memory
    """

    def __init__(self, associative_memory, matrix_y, matrix_x, size_x):
        """Constructs all the necessary attributes for the associative memory.

        Args:
            associative_memory::str(="hopfield", "hopfield_hysterisis", "pcsmn_1",
            "pcsmn_2", "s_gcm", "s_gcm_2", "cl_gcm", "si_gcm", "fam_zhang", "fam_xiao",
            "fam_chunglee", "fam_liu", "mam_max", "mam_plus_max", "mam_fuzzy_max",
            "mam_rev_fuzzy_max", "mam_log_max", "ifam_godel_max", "ifam_goguen_max",
            "ifam_luka_max", "exor_max", "mam_min", "mam_plus_min", "mam_fuzzy_min",
            "mam_rev_fuzzy_min", "mam_log_min", "ifam_godel_min", "ifam_goguen_min",
            "ifam_luka_min", "exor_min", "kernel_exor" or "kernel_mam_heuristic")
                associative memory codename.
            pattern_x::array object
                Matrix of column vectors corresponding to the input patterns
                of the training set.
            pattern_y::array object
                Matrix of column vectors corresponding to the output patterns
                of the training set.
            size_x::tuple(2)(rows,columns)
                Size of the kernel images to be saved (rows,columns).
                rows::int
                    Pixel value of the number of image rows (Height).
                columns::int
                    Pixel value of the number of image columns (Width).

        Returns:
            None
        """

        self.associative_memory = associative_memory
        self.matrix_x = matrix_x
        self.matrix_y = matrix_y
        self.size_x = size_x

        # Matrix recalled by associative memory
        self.matrix_recall = np.array([], dtype = 'float32')

        if self.associative_memory not in ["kernel_mam_heuristic", "kernel_exor"]:
            # Associative memory learning matrix
            self.matrix_learning = np.array([], dtype = 'float32')

            if self.associative_memory in ["ifam_godel_max", "ifam_goguen_max", "ifam_luka_max"]:
                self.theta_out = np.max(self.matrix_y, axis = 1)
                self.theta_out = self.theta_out.reshape((self.matrix_y.shape[0], 1))

            if self.associative_memory in ["ifam_godel_min", "ifam_goguen_min", "ifam_luka_min", "fam_liu"]:
                self.theta_out = np.min(self.matrix_y, axis = 1)
                self.theta_out = self.theta_out.reshape((self.matrix_y.shape[0], 1))

                if self.associative_memory == "fam_liu":
                    matrix_a = copy.deepcopy(self.matrix_x).astype(np.float32)
                    matrix_b = copy.deepcopy(self.matrix_y).astype(np.float32)
                    matrix_l_e = np.less_equal(matrix_a[:, :, None], matrix_b.transpose())
                    theta = np.zeros((1, matrix_a.shape[0]))

                    for index in np.ndindex( theta.shape[1] ):
                        i = index[0]
                        if np.count_nonzero(matrix_l_e[i, :, :]) > 0:
                            coord = np.where(matrix_l_e[i, :, :])
                            coord = [coord[1], coord[0]]
                            theta[0, i] = np.min(matrix_b[tuple(coord)])

                    self.theta_in = theta.transpose()
        else:
            # Kernel learning matrix M_ZZ
            self.matrix_out_mzz = np.array([], dtype='float32')

            if self.associative_memory == "kernel_mam_heuristic":
                self.matrix_z= kernel_heuristic(self.matrix_y, self.matrix_x, self.size_x)
            if self.associative_memory == "kernel_exor":
                self.matrix_z = kernel_hattori(self.matrix_y, self.matrix_x, self.size_x)

            # Calculates matrix M_ZZ of kernel
            name = associative_memories[self.associative_memory]["memory_mzz"]
            self.memory_mzz = AssociativeMemory(name, self.matrix_z, self.matrix_z, self.size_x)

            # Calculates matrix W_ZY of kernel
            name = associative_memories[self.associative_memory]["memory_wzy"]
            self.memory_wzy = AssociativeMemory(name, self.matrix_y, self.matrix_z, self.size_x)


    def learn(self):
        """Constructs associative memory learn matrix.

        Args:
            None.
        Returns:
            None
        """

        if self.associative_memory not in ["kernel_mam_heuristic", "kernel_exor"]:
            if self.associative_memory in ["mam_max", "mam_min"]:
                self.matrix_learning = learn_morphological(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_plus_max", "mam_plus_min"]:
                self.matrix_learning = learn_morphological_plus(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_fuzzy_max", "mam_fuzzy_min"]:
                self.matrix_learning = learn_morphological_fuzzy(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_rev_fuzzy_max", "mam_rev_fuzzy_min"]:
                self.matrix_learning = learn_morphological_reverse_fuzzy(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_log_max", "mam_log_min"]:
                self.matrix_learning = learn_morphological_logarithmic(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["ifam_godel_max", "ifam_goguen_max", "ifam_luka_max", "ifam_godel_min", "ifam_goguen_min", "ifam_luka_min"]:
                self.matrix_learning = learn_implicative(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["exor_max", "exor_min"]:
                self.matrix_learning = learn_exor(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["hopfield", "hopfield_hysterisis"]:
                self.matrix_learning = learn_hopfield(self.matrix_y, self.matrix_x)
            if self.associative_memory in ["s_gcm", "s_gcm_2", "cl_gcm", "si_gcm"]:
                self.matrix_learning = learn_gcm(self.matrix_y, self.matrix_x)
            if self.associative_memory in ["pcsmn_1", "pcsmn_2"]:
                self.matrix_learning = learn_pcsmn(self.matrix_y, self.matrix_x)
            if self.associative_memory in ["fam_zhang", "fam_chunglee", "fam_xiao"]:
                self.matrix_learning = learn_fuzzy(self.matrix_y, self.matrix_x, self.associative_memory)
            if self.associative_memory in ["fam_liu"]:
                self.matrix_learning = learn_fuzzy_liu(self.matrix_y, self.matrix_x)
        else:
            self.memory_mzz.learn()
            self.memory_wzy.learn()


    def recall(self, matrix_x):
        """Recall pattern by associative memory.

        Args:
            pattern::array object
                Matrix that is presented to the associative memory in order to
                recall associative memory patterns.
        Returns:
            matrix_out::array object
                Matrix recalled by associative memory
        """

        if self.associative_memory not in ["kernel_mam_heuristic", "kernel_exor"]:
            if self.associative_memory in ["mam_max", "mam_min"]:
                self.matrix_recall = recovery_morphological(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_plus_max", "mam_plus_min"]:
                self.matrix_recall = recovery_morphological_plus(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_fuzzy_max", "mam_fuzzy_min"]:
                self.matrix_recall = recovery_morphological_fuzzy(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_rev_fuzzy_max", "mam_rev_fuzzy_min"]:
                self.matrix_recall = recovery_morphological_reverse_fuzzy(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["mam_log_max", "mam_log_min"]:
                self.matrix_recall = recovery_morphological_logarithmic(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["ifam_godel_max", "ifam_goguen_max", "ifam_luka_max"]:
                self.matrix_recall = recovery_implicative(self.matrix_learning, matrix_x, self.associative_memory)
                self.matrix_recall = np.minimum(self.matrix_recall, self.theta_out)
            if self.associative_memory in ["ifam_godel_min", "ifam_goguen_min", "ifam_luka_min"]:
                self.matrix_recall = recovery_implicative(self.matrix_learning, matrix_x, self.associative_memory)
                self.matrix_recall = np.maximum(self.matrix_recall, self.theta_out)
            if self.associative_memory in ["exor_max", "exor_min"]:
                self.matrix_recall = recovery_exor(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["hopfield"]:
                self.matrix_recall = recovery_hopfield(self.matrix_learning, matrix_x)
            if self.associative_memory in ["hopfield_hysterisis"]:
                self.matrix_recall = recovery_hopfield(self.matrix_learning, matrix_x, hysterisis = True)
            if self.associative_memory in ["s_gcm", "s_gcm_2", "cl_gcm", "si_gcm"]:
                self.matrix_recall = recovery_gcm(self.matrix_learning, matrix_x, self.matrix_y, self.associative_memory)
            if self.associative_memory in ["pcsmn_1", "pcsmn_2"]:
                self.matrix_recall = recovery_pcsmn(self.matrix_learning, matrix_x, self.matrix_y, self.associative_memory)
            if self.associative_memory in ["fam_zhang", "fam_chunglee", "fam_xiao"]:
                self.matrix_recall = recovery_fuzzy(self.matrix_learning, matrix_x, self.associative_memory)
            if self.associative_memory in ["fam_liu"]:
                matrix_temp_x = np.maximum(matrix_x, self.theta_in)
                self.matrix_recall = recovery_fuzzy_liu(self.matrix_learning, matrix_temp_x)
                self.matrix_recall = self.matrix_recall.transpose()
                self.matrix_recall = np.maximum(self.matrix_recall, self.theta_out)


                #matrix_x = np.maximum(matrix_x, self.theta_in)
                #self.matrix_out = recovery_fuzzy_liu(self.matrix_out, matrix_x)
                #self.matrix_out = self.matrix_out.transpose()
                #self.matrix_out = np.maximum(self.matrix_out, self.theta_out)

        else:
            self.matrix_out_mzz = self.memory_mzz.recall(matrix_x)
            self.matrix_recall = self.memory_wzy.recall(self.matrix_out_mzz)

        self.matrix_recall[self.matrix_recall < 0] = 0
        self.matrix_recall[self.matrix_recall > 1] = 1

        return self.matrix_recall