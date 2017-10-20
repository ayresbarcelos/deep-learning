import tests.project2.helper as helper
import tests.project2.problem_unittests as tests
import numpy as np
import matplotlib.pyplot as plt

cifar10_dataset_folder_path = '../../image-classification/cifar-10-batches-py'

def downloadImages():
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    from urllib.request import urlretrieve
    from os.path import isfile, isdir
    from tqdm import tqdm
    import tests.project2.problem_unittests as tests
    import tarfile

    cifar10_dataset_folder_path = '../../image-classification/cifar-10-batches-py'

    # Use Floyd's cifar-10 dataset if present
    floyd_cifar10_location = '../../image-classification/cifar-10-python.tar.gz'
    if isfile(floyd_cifar10_location):
        tar_gz_path = floyd_cifar10_location
    else:
        tar_gz_path = 'cifar-10-python.tar.gz'

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(tar_gz_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_gz_path,
                pbar.hook)

    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open(tar_gz_path) as tar:
            tar.extractall()
            tar.close()

    tests.test_folder_path(cifar10_dataset_folder_path)


# downloadImages()

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    return x / 255.


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    result = np.zeros([len(x), 10])
    for i, lbl in enumerate(x):
        result[i][lbl] = 1
    return result


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)

if __name__ == '__main__':

    # Explore the dataset
    batch_id = 1
    sample_id = 5
    helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)

    # Preprocess Training, Validation, and Testing Data
    helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

    import pickle
    import problem_unittests as tests
    import helper

    # Load the Preprocessed Validation data
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))