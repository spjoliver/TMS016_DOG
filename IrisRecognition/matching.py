import numpy as np

def hamming_distance_absolute(iris_code1, iris_code2):
    if len(iris_code1) != len(iris_code2):
        raise ValueError("IrisCodes must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(iris_code1, iris_code2))


def hamming_distance_percentage(code1, code2):
    if len(code1) != len(code2):
        raise ValueError("IrisCodes must be of the same length")
        
    mismatches = sum(c1 != c2 for c1, c2 in zip(code1, code2))
    total_length = len(code1)
    
    # calculate percentage of mismatches
    percentage = (mismatches / total_length) * 100
    
    return percentage

def hamming_distance_absolute_flatten(iris_code1, iris_code2):
    iris_code1_flat = np.array(iris_code1).flatten()
    iris_code2_flat = np.array(iris_code2).flatten()
    
    if len(iris_code1_flat) != len(iris_code2_flat):
        raise ValueError("IrisCodes must be of the same length")
    
    return sum(c1 != c2 for c1, c2 in zip(iris_code1_flat, iris_code2_flat))


def hamming_distance(a, b):
    return np.count_nonzero(a != b)


def shifted_hamming(a, b):
    # Function to calculate Hamming distance
    def hamming_distance(a, b):
        return np.count_nonzero(a != b)
    
    distances = []
    for i in range(b.shape[1]):  # for each column
        shifted_b = np.roll(b, shift=i, axis=1)  # shift columns of b by i
        distances.append(hamming_distance(a, shifted_b))

    return distances




def shifted_hamming_all_values(a, b):
    # Function to calculate Hamming distance
    def hamming_distance(a, b):
        return np.count_nonzero(a != b)
    
    a = a.ravel()  # Flatten a
    b = b.ravel()  # Flatten b

    distances = []
    for i in range(len(b)):  # for each element
        shifted_b = np.roll(b, shift=i)  # shift elements of b by i
        distances.append(hamming_distance(a, shifted_b))

    return distances




def shifted_hamming_all_values_increments(a, b):
    # Function to calculate Hamming distance
    def hamming_distance(a, b):
        return np.count_nonzero(a != b)
    
    original_shape = a.shape  # Save original shape
    cols = original_shape[1]  # Number of columns

    a = a.ravel()  # Flatten a
    b = b.ravel()  # Flatten b

    distances = []
    for i in range(original_shape[0]):  # for each row
        shifted_b = np.roll(b.reshape(original_shape), shift=i*cols).ravel()  # shift elements of b by i rows and flatten again
        distances.append(hamming_distance(a, shifted_b))

    return distances
