import numpy as np
from scipy.optimize import fmin_cg

class NN(object):
  def __init__(self, X, y, neurons, num_labels):
    self.X = X
    self.m, self.n = X.shape                                            # [m n] = [ shape[0] shape[1] ]
    self.layers = len(neurons)                                          # total number of layers in the network

    if self.layers < 3:
      raise ValueError("Must have an input, a hidden and an output layer.")

    self.neurons = neurons                                              # neurons
    self.in_neurons = neurons[0]                                        # number of input neurons
    self.out_neurons = neurons[-1]                                      # number of output neurons
    self.h_neurons = neurons[1:-1]                                      # total number of hidden neurons
    self.h_layers = len(self.h_neurons)                                 # number of hidden layers

    if self.in_neurons != self.n:
      raise ValueError("Number of features and number of input units are not same.")

    self.Theta = np.array([None for i in range(self.layers-1)])         # (layers-1) theta matrices

    self.yv = (y == num_labels).astype('int')      # because just y is of no use


  # to generate random numbers in [-epsilon, +epsilon]
  @staticmethod
  def generateEpsilon(L_in, L_out):
    # sqrt(6) ~ 2.45
    return round(2.45 / ((L_in + L_out) ** .5), 2)


  # returns the sigmoid value of z vector
  # sigmoid function: g(x) = 1 / (1 + exp(-x))
  @staticmethod
  def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


  # rolls the unrolled vector according to units tuple/array in following manner
  # each matrix of size units[i+1] * (units[i]+1)
  # number of matrix = number of layers - 1
  @staticmethod
  def rollVector(vec, units):
    layers = len(units)
    start = 0
    Matrix = np.array([None for i in range(layers-1)])
    for i in range(layers-1):
      Matrix[i] = np.reshape(vec[start : start+(units[i+1] * (units[i]+1))], (units[i+1], units[i]+1))
      start = units[i+1] * (units[i]+1)

    return Matrix


  # as the name suggests and the code does!!
  @staticmethod
  def unrollVec(vec):
    return np.hstack(tuple(matrix.flatten() for matrix in vec))


  # initialize Theta with random values to break symmetry
  def initializeTheta(self):
    for i in range(self.layers-1):
      epsilon_init = self.generateEpsilon(L_in=self.neurons[i], L_out=self.neurons[i+1])
      self.Theta[i] = np.random.rand(self.neurons[i+1], self.neurons[i]+1) * (2 * epsilon_init) - epsilon_init


  # performs forward propagation on NN
  # takes in already rolled parameter: Theta because it is called from compute_cost & compute_gradient
  # returns all activation layers -- increases re-usability
  def for_prop(self, Theta):

    a = np.array([None for i in range(self.layers)])                  # activation units/neurons
    a[0] = self.X.astype('float64')                                   # initializing a(0)

    for i in range(self.layers-1):
      a[i] = np.concatenate((np.ones((len(a[i]), 1)), a[i]), axis=1)   # concatenating column-wise
      a[i+1] = self.sigmoid(a[i].dot(Theta[i].T))                      # a(i+1) = g( z(i) )

    return a     # return all activation layers


  # cost function of the neural network
  # J(\Theta) = sum(sum(yv .* log(h) + (1-yv) .* log(1-h))) / (-m)
  # reg_lambda: the regularisation parameter
  def compute_cost(self, params, *args):
    reg_lambda = args[0]

    # rolling
    Theta = self.rollVector(params, self.neurons)

    h = self.for_prop(Theta)[self.layers-1]

    J = np.sum(np.sum(self.yv * np.log(h) + (1-self.yv) * np.log(1-h)), axis=0) / (-self.m)

    if reg_lambda != 0:
      weights = 0
      for theta in Theta:
        weights += np.sum(np.sum(theta[:, 1:] ** 2, axis=0))

      J += weights * (reg_lambda / (2*self.m))

    return J


  # computes partial derivatives of the cost function of NN
  # using backward propagation algorithm
  # reg_lambda: the regularisation parameter
  def compute_gradient(self, params, *args):
    reg_lambda = args[0]

    # useful variables
    Theta = self.rollVector(params, self.neurons)                       # rolling
    d = np.array([None for i in range(self.layers)])                    # d[0] = None
    Delta = np.array([None for i in range(self.layers-1)])              # accumulator
    theta_grad = np.array([None for i in range(self.layers-1)])         # partial derivatives

    a = self.for_prop(Theta)

    # computing errors
    d[self.layers-1] = a[self.layers-1] - self.yv
    for i in range(self.layers-2, 0, -1):                                # assumption: at least one hidden layer
      sigmoid_gradient = a[i][:, 1:] * (1 - a[i][:, 1:])
      d[i] = (d[i+1].dot(Theta[i][:, 1:])) * sigmoid_gradient

    # computing Delta
    for i in range(self.layers-1):
      Delta[i] = np.dot(d[i+1].T, a[i])

    # computing partial derivatives
    for i in range(self.layers-1):
      theta_grad[i] = (Delta[i] / self.m)

    # regularising
    for i in range(self.layers-1):
      theta_grad[i][:, 1:] += (reg_lambda / self.m) * Theta[i][:, 1:]

    return self.unrollVec(theta_grad)


  # WARNING: Must initialize theta first then train the neural network
  # trains the neural network using the fmin_cg optimization algorithm
  # maxIter: maximum number of iterations
  # returns a tuple: (optimized_cost, optimized_rolled_Theta)
  def train(self, maxIter=None, reg_lambda=0):
    args = (reg_lambda,)
    res = fmin_cg(f=self.compute_cost, x0=self.unrollVec(self.Theta), fprime=self.compute_gradient,
                  maxiter=maxIter, full_output=True, args=args)
    optimized_theta = self.rollVector(res[0], self.neurons)
    optimized_cost = res[1]
    return optimized_cost, optimized_theta



# example
if __name__ == '__main__':
  from scipy.io import loadmat

  X = loadmat('dummy_data\\ex4data1.mat')['X']
  y = loadmat('dummy_data\\ex4data1.mat')['y']

  myNeuralNetwork = NN(X, y, (400, 25, 10), num_labels=np.arange(1, 10+1))
  myNeuralNetwork.initializeTheta()
  result = myNeuralNetwork.train(maxIter=50, reg_lambda=3)

  print("Optimized Cost = %f" % result[0])
  for i in range(len(result[1])):
    print("Size of Theta[%d] = %s" % (i, str(result[1][i].shape)))
