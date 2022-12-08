"""
scripts.parameters
----------------------------
This module contains in dictionary form the main attributes of the associative
memories and working directory paths used in the code.
"""

__author__ = "Arturo Gamino-Carranza"
__version__ = "1.0.0"
__email__ = "arturogamino@hotmail.com"

import os

offset = 5

associative_memories = {
    "hopfield": {
        "id": 0,
        "name": "Hopfield",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "hopfield_hysterisis": {
        "id": 1,
        "name": "Hopfield with hysterisis",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "pcsmn_1": {
        "id": 2,
        "name": "Parametrically coupled sine map network 1 (PCSMN 1)",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "pcsmn_2": {
        "id": 3,
        "name": "Parametrically coupled sine map network 2 (PCSMN 2)",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "s_gcm": {
        "id": 4,
        "name": "Globally coupled map using the symmetric map (S-GCM)",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "s_gcm_2": {
        "id": 5,
        "name": "A new parameter control method for S-GCM",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "cl_gcm": {
        "id": 6,
        "name": "Globally coupled map using cubic logistic map (CL-GCM)",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "si_gcm": {
        "id": 7,
        "name": "Globally coupled map using sine map (SI-GCM)",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative"]
    },
    "fam_zhang": {
        "id": 8,
        "name": "Improved model of optical fuzzy associative memory",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "fam_xiao": {
        "id": 9,
        "name": "Max-min encoding learning algorithm for fuzzy max-multiplication",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "fam_chunglee": {
        "id": 10,
        "name": "Fuzzy relational memory",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "fam_liu": {
        "id": 11,
        "name": "Max-min fuzzy neural network with threshold",
        "noise": ["additive", "subtractive", "mixed"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_max": {
        "id": 12,
        "name": "Morphological associative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_plus_max": {
        "id": 13,
        "name": "New method of morphological associative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_fuzzy_max": {
        "id": 14,
        "name": "New fuzzy morphological associative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_rev_fuzzy_max": {
        "id": 15,
        "name": "No rounding reverse fuzzy morphological associative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_log_max": {
        "id": 16,
        "name": "Logarithmic and exponential morphological associative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "ifam_godel_max": {
        "id": 17,
        "name": "Godel fuzzy implicative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "ifam_goguen_max": {
        "id": 18,
        "name": "Goguen fuzzy implicative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "ifam_luka_max": {
        "id": 19,
        "name": "Lukasiewicz fuzzy implicative memory type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "exor_max": {
        "id": 20,
        "name": "Associative memory with complemented operations type max",
        "noise": ["additive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_min": {
        "id": 21,
        "name": "Morphological associative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_plus_min": {
        "id": 22,
        "name": "New method of morphological associative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_fuzzy_min": {
        "id": 23,
        "name": "New fuzzy morphological associative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_rev_fuzzy_min": {
        "id": 24,
        "name": "No rounding reverse fuzzy morphological associative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "mam_log_min": {
        "id": 25,
        "name": "Logarithmic and exponential morphological associative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "ifam_godel_min": {
        "id": 26,
        "name": "Godel fuzzy implicative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "ifam_goguen_min": {
        "id": 27,
        "name": "Goguen fuzzy implicative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "ifam_luka_min": {
        "id": 28,
        "name": "Lukasiewicz fuzzy implicative memory type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "exor_min": {
        "id": 29,
        "name": "Associative memory with complemented operations type min",
        "noise": ["subtractive"],
        "mode": ["autoassociative", "heteroassociative"]
    },
    "kernel_exor": {
        "id": 30,
        "name": "Kernel for associative memory with complemented operations",
        "noise": ["mixed"],
        "memory_mzz": "exor_max",
        "memory_wzy": "exor_min",
        "mode": ["autoassociative", "heteroassociative"]
    },
    "kernel_mam_heuristic": {
        "id": 31,
        "name": "Heuristic kernel for morphological associative memory",
        "noise": ["mixed"],
        "memory_mzz": "mam_max",
        "memory_wzy": "mam_min",
        "mode": ["autoassociative", "heteroassociative"]
    }
}

"""A dictionary of the directory paths where the training patterns, noisy patterns,
kernels and simulation results are stored
"""
paths_directories = {
    "training_input_pattern": os.path.join(os.getcwd(), "dataset/input/"),
    "training_output_pattern": os.path.join(os.getcwd(), "dataset/output/"),
    "input_pattern": os.path.join(os.getcwd(), "simulation/pattern/input/"),
    "output_pattern": os.path.join(os.getcwd(), "simulation/pattern/output/"),
    "kernel_pattern": os.path.join(os.getcwd(), "simulation/pattern/kernel/"),
    "noise_pattern": os.path.join(os.getcwd(), "simulation/noise/"),
    "recall_pattern": os.path.join(os.getcwd(), "simulation/recall/")
}