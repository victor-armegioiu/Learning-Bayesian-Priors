import numpy as np
import sklearn


class UnknownDataGenerationError(Exception):
  pass


def GetSinusoidParams():
  """Parameters for fa,b,c,β(x) = β ∗ x + a ∗ sin(1.5 ∗ (x − b)) + c.
  """
  a = np.random.uniform(0.7, 1.3)
  b = np.random.normal(0, 0.1)
  c = np.random.normal(5, 0.1)
  beta = np.random.normal(0.5, 0.2)

  return a, b, c, beta


def GetDataset(config):
  """Generate synthetic data for small regression problems.

  Given a configuration of the required synthetic
  data this method currently supports:
      - Generation of regressors from an uniform distribution, where
        regressands are computed as β ∗ x + a ∗ sin(1.5 ∗ (x − b)) + c.
      - Generating a random regression problem.
  
  """

  generation_method = config['generation_method']
  meta_task_cnt = config['meta_training_task_cnt']
  meta_tuning_task_cnt = config['meta_tuning_task_cnt']

  task_size = config['task_size']
  input_shape = config['input_shape']
  all_training_tasks = []

  if 'sin' in generation_method:
    for _ in range(meta_task_cnt + meta_tuning_task_cnt):
      a, b, c, beta = GetSinusoidParams()
      x = np.sort(np.random.uniform(-5, 5, task_size))[:, None]
      y = beta * x + a * np.sin(1.5 * (x - b)) + c
      all_training_tasks.append((x, y))

  elif 'sklearn' in generation_method:
    x, y = sklearn.datasets.make_regression(size, input_shape)
    all_training_tasks.append((x, y))
      
  else:
    raise UnknownDataGenerationError

  return (all_training_tasks[:-meta_tuning_task_cnt],  
          all_training_tasks[-meta_tuning_task_cnt:])
