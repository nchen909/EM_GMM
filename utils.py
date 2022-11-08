import numpy as np

def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())

def learn_params(x_labeled, y_labeled):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    params = {'phi': None, 'mu0': None, 'mu1': None, 'sigma0': None, 'sigma1': None}
    return params