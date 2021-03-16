import numpy as np
import gpflow
import tensorflow as tf
from tensorflow.python.keras import backend as K

K.set_floatx("float64")


def _GetMeanFunction():
  inputs = tf.keras.layers.Input(shape=(1,))
  x = tf.keras.layers.Dense(64, activation="relu")(inputs)
  x = tf.keras.layers.Dense(64, activation="relu")(x)
  outputs = tf.keras.layers.Dense(1)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def _GetModel(data, mean_function, kernel):
  model = gpflow.models.GPR(data, kernel=kernel, mean_function=mean_function)
  model.likelihood.variance.assign(1e-2)
  return model


def _GetOptimizationStep(optimizer, model: gpflow.models.GPR):
    @tf.function
    def optimization_step():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            objective = model.training_loss()
        grads = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return objective
    return optimization_step


def _RunAdam(model, iterations):
    """
    Utility function for running the Adam optimizer.

    @params: 
      model: GPflow model.
      iterations: number of iterations to run Adam for on this task.
    """
    logf = []
    adam = tf.optimizers.Adam()
    optimization_step = _GetOptimizationStep(adam, model)
    for step in range(iterations):
        loss = optimization_step()
        if step % 10 == 0:
            logf.append(loss.numpy())
    return logf

def TrainGPPrior(meta_tasks,
                num_iter=5,
                num_iter_per_task=5,
                verbose=False,
                print_every_n=5):
  """
  Metalearning training loop.

  @params:
      meta_tasks: list of metatasks.
      num_iter: number of iterations/epochs on tasks set.
      num_iter_per_task: number of iterations per tasks.
      print_every_n: number of tasks between printing 
        likelihood estimates.
  @returns:
    mean_function: trained mean function neural net.
  """
  # Initialize mean function neural net and kernel.
  mean_function = _GetMeanFunction()
  kernel = gpflow.kernels.Matern52()

  # Iterate for several passes over the tasks set.
  for iteration in range(num_iter):
    print("\nCurrently in meta-iteration/epoch {}/{}:".format(iteration + 1,
                                                              num_iter))
    for i, task in enumerate(meta_tasks):
      data = task  # (X, Y)
      model = _GetModel(data, kernel=kernel, mean_function=mean_function)
      _ = _RunAdam(model, num_iter_per_task)
      
      if verbose and i % print_every_n == 0:
        print('Task {} log-likelihood: {}'.format(i, 
                                          model.log_marginal_likelihood()))
          
  return mean_function, kernel
