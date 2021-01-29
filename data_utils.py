import numpy as np
import sklearn


class UnknownDataGenerationError(Exception):
    pass


def GetSinusoidParams(task_count):
    """Parameters for fa,b,c,β(x) = β ∗ x + a ∗ sin(1.5 ∗ (x − b)) + c.

    Regressand is calculated as a function of the regressor
    and the given fixed params.

    @params:
        task_count: Number of sets of hyperparameters to reproduce.
    """
    a = np.random.uniform(0.7, 1.3, task_count)
    b = np.random.normal(0, 0.1, task_count)
    c = np.random.normal(5, 0.1, task_count)
    beta = np.random.normal(5, 0.2, task_count)

    tasks_data = []
    for i in range(task_count):
        task_data = {'a': a[i], 'b': b[i], 'c': c[i], 'beta': beta[i]}
        tasks_data.append(task_data)
    return tasks_data


def GetDataset(config):
    """Generate synthetic data for small regression problems.

    Given a configuration of the required synthetic
    data this method currently supports:
        - Generation of regressors from an uniform distribution, where
          regressands are computed as β ∗ x + a ∗ sin(1.5 ∗ (x − b)) + c.
        - Generating a random regression problem.
    """

    generation_method = config['generation_method']
    size = config['size']
    input_shape = config['input_shape']

    x_train, y_train = None, None
    if 'sin' in generation_method:
        tasks_data = config['tasks_data']
        for task_data in tasks_data:
            a, b, c, beta = (task_data['a'], task_data['b'],
                             task_data['c'], task_data['beta'])
            x = np.random.uniform(-5, 5, size)
            y = beta * x + a * np.sin(1.5 * (x - b)) + c

            if x_train is None:
                x_train = x
                y_train = y
            else:
                x_train = np.concatenate((x_train, x))
                y_train = np.concatenate((y_train, y))

    elif 'sklearn' in generation_method:
        x_train, y_train = sklearn.datasets.make_regression(size,
                                                            input_shape)

    else:
        raise UnknownDataGenerationError

    return x_train, y_train