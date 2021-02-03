
import numpy as np
import tensorflow as tf

def GetSlicedScoreEstimator(config, print_every_n=10, verbose=False):
  """Train a NN to perform score estimation based on samples.

  @params:
    config: dict, contains training setting, the samples and the model to be  
    optimized.

    print_every_n: number of epochs to wait before printing loss info.
    verbose: Printing is done only if this is set to `True`.
  """
  x = config['data']
  score_net = config['score_net']

  # Training setup.
  epochs = config['epochs']
  lambda_reg = config['lambda_reg']
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  best_model = tf.keras.models.clone_model(score_net)
  best_loss = 1e9

  x_tensor = tf.cast(x, tf.float64)
  for epoch in range(epochs):
    with tf.GradientTape(persistent=True) as g:
      g.watch(x_tensor)
      vectors = tf.random.normal(shape=x_tensor.shape, dtype=tf.float64)
      
      grad1 = tf.cast(score_net(x_tensor), tf.float64)
      gradv = tf.math.reduce_sum(grad1 * vectors, axis=0)

      loss1 = lambda_reg * tf.math.reduce_sum(
          (grad1 * vectors) ** 2, axis=0) 

      grad2 = g.gradient(gradv, x_tensor)
      loss2 = tf.math.reduce_sum(vectors * grad2, axis=0)

      # Combine the losses.
      loss_val = tf.reduce_mean(loss1 + loss2)

    grads = g.gradient(loss_val, score_net.trainable_weights)
    optimizer.apply_gradients(zip(grads, score_net.trainable_weights))

    if loss_val.numpy() < best_loss:
      best_loss = loss_val.numpy()
      best_model.set_weights(score_net.get_weights())

    if verbose and epoch % print_every_n == 0:
      print('Epoch [%d], loss: [%f]' % (epoch, loss_val.numpy(),))

  return best_model, best_loss
