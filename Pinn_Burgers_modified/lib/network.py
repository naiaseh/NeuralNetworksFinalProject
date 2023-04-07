import tensorflow as tf
from keras import backend as K
from functools import partial 

# Define the activation function
def tanh_activation(x, a):
    return (K.tanh(a*x*5))

class Network:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.
    """
    def __init__(self, a = 0.34):
        """
        Initialize the PINN model.

        Args:
            a: initial value of the trainable slope parameter.
        """
        self.A = tf.Variable(initial_value=a, trainable=True, dtype=tf.float32)

    def build(self, num_inputs=2, layers=[20, 20, 20, 20, 20, 20], num_outputs=1):
        """
        Build a PINN model for Burgers' equation with input shape (t, x) and output shape u(t, x).

        Args:
            num_inputs: number of input variables. Default is 2 for (t, x).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 1 for u(t, x).

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        # hidden layers
   
        x = inputs
        for layer in layers:
            x = tf.keras.layers.Dense(layer,
                kernel_initializer='he_normal')(x)
            x = tf.keras.layers.Activation(partial(tanh_activation,a=self.A))(x)
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)

