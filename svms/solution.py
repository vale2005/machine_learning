import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(100)

# samples
classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [1.0, 1.0], np.random.randn(10, 2) * 1.0 + [1.5, 1.5]))
classB = np.random.randn(20, 2) * 1.0 + [0.5, 0.5]
inputs = np.concatenate((classA, classB))

targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

# randomize them
N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

#plot the samples
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p [ 0 ] for p in classB], [p[1] for p in classB ], 'r.')
plt.axis('equal')

# kernel functions Functions
def linear_kernel_fun(point1, point2):
    return np.dot(point1, point2)

def polinomyal_kernel_fun(point1, point2, P=5.0):
    dotted = np.dot(point1, point2) + 1.0
    return np.power(dotted, P)

def rbf_kernel(point1, point2, SIGMA = 70.0):
    norm = np.linalg.norm(point1-point2)
    exponent = -pow(norm, 2)/(SIGMA*pow(1.0, 2.0))
    return math.exp(exponent)


# precompute values so minimization is faster
def precompute_mtx():
    vec = np.array([targets[i]*targets[j]*KERNEL(inputs[i], inputs[j]) for i in range(0, N) for j in range(0, N)])
    return np.reshape(vec, (N, N))


# function to minimize, basically the dual formulation
def objective_fun(vec):
    temp = np.dot(vec, precomputed_mtx)
    ret = np.dot(temp, vec)
    alpha_sum = np.sum(vec)
    return 0.5 * np.sum(ret) - alpha_sum


# constraint for minimization
def zero_fun(vec):
    return np.dot(vec, targets)


def solve_minimization(start_vector):
    XC = {'type':'eq', 'fun':zero_fun}
    ret = minimize(objective_fun, start_vector, bounds=bounds, constraints=XC)
    if ret["success"]:    
        alphas = ret['x']
        print("Alphas: %s " % alphas)
        non_zero = non_zero_alphas(alphas)
        b = b_value(non_zero)
        return non_zero,b
    else:
        print("No solution")


# get alpha values on the margin
def non_zero_alphas(alphas):
    threshold = pow(10, -5)
    non_zero = [[alpha_val, inputs[index], targets[index]] for index, alpha_val in enumerate(alphas) if alpha_val > threshold]
    return non_zero


# get the bias
def b_value(alphas):
    alpha_value = alphas[0][1]
    b_value = np.array([vec[0]*vec[2]*KERNEL(alpha_value, vec[1]) for vec in alphas])
    return np.sum(b_value) - alphas[0][2]


# classify new points
def indicator(vector, b, s_value):
    indicator = np.array([vec[0]*vec[2]*KERNEL(s_value, vec[1]) for vec in vector])
    return np.sum(indicator) - b


def draw_plot(alpha, b):
    xgrid=np.linspace(-5, 5)
    ygrid=np.linspace (-5, 5)
    grid=np.array([[indicator(alpha, b, (x, y)) for x in xgrid] for y in ygrid])
    plt.contour(xgrid, ygrid, grid ,(-1.0, 0.0, 1.0 ), colors = ('red', 'black', 'blue'), linewidths =(1, 3, 1))
    plt.show()


KERNEL = rbf_kernel
SLACK = 8.0
bounds = [(0, SLACK) for i in range(0, N)]

precomputed_mtx = precompute_mtx()
alpha, b = solve_minimization(np.zeros(N))
draw_plot(alpha, b)