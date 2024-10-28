# NeuralNetworksFinalProject

This code is taken from [Okada39](https://github.com/okada39/pinn_burgers) and modified for a tanh(anx) custom activation function, where $n$ is fixed and $a$ is the slope. Here are a list of other changes in addition to the activation function:
* Mode analysis - note that the modes look inverted. Meaning that if they were transformed with $F(k) = {\Sigma_{n}}^{N-1} e^{i2\pi \frac{kn}{N}} u(t',x_n)$ instead of $F(k) = {\Sigma_{n}}^{N-1} e^{-i2\pi \frac{kn}{N}} u(t',x_n)$ we would have the correct frequencies. However, this shouldn't happen as we tested with the some analytics sinusoidal series and the transformations were reasonable. 
* Loss was made an attribute of the optimizer
* Number of hidden layers and neurons changed to that of the paper (i.e. 6 layers with 20 neurons each)

