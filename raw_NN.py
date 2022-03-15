import numpy as np
import sys
import random
import json

#TO DO
#implement a __call__() method
#code in other cost functions or activation functions
#for other activation functions, must make function an argument
#in classes like QuadraticCost and CrossEntropyCost



def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

class QuadraticCost():

    #Quadratic cost function functions    

    @staticmethod
    def func(x, y):
        return 0.5*np.linalg.norm(x - y)**2
    
    @staticmethod
    def err(z, x, y):
        #error in output layer
        return (x - y)*sigmoid_prime(z)

class CrossEntropyCost():
    
    #Cross entropy cost function functions
    
    @staticmethod
    def func(x, y):
        return np.sum(np.nan_to_num((-y*np.log(x) - (1 - y)*np.log(1 - x)))) 

    @staticmethod
    def err(z, x, y):
        #error in output layer
        return (x - y)
    
class Network():
    
    def __init__(self, sizes, cost=CrossEntropyCost):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost
        
    def default_weight_initializer(self):
        self.biases = [np.random.normal(size=(x, 1)) for x in self.sizes[1:]]
        self.weights = [np.random.normal(x, y)/np.sqrt(x) 
                        for y, x in zip(self.sizes[:-1], self.sizes[1:])]
    
    def large_weight_initializer(self):
        self.biases = [np.random.normal(size=(x, 1)) for x in self.sizes[1:]]
        self.weights = [np.random.normal(size=(x, y)) 
                        for y, x in zip(self.sizes[:-1], self.sizes[1:])]
    
    def feedforward(self, a):
        #Input is a
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        #Train the neural network
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            #shuffle rows of training data
            random.shuffle(training_data)
            #Create a partitioned version of training_data, where 
            #each partition is a mini batch. mini_batches is a matrix.
            mini_batches = [training_data[k:k + mini_batch_size] 
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost  = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: \
                      {self.accuracy(evaluation_data)} / {n_data}")
            return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
        
        def update_mini_batch(self, mini_batch, eta, lmbda, n):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in mini_batch:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta/len(mini_batch))*nb
                           for b, nb in zip(self.biases, nabla_b)]
            
        def backprop(self, x, y):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            
            activation = x
            activations = [x] #Each layer's activations
            zs = [] #Each layer's weighted input
            
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            
            delta = (self.cost).err(zs[-1], activations[-1], y)
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            
            for l in range(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l + 1].transpose(), delta)*sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            return (nabla_b, nabla_w)
    
        def accuracy(self, data, convert=False):
            if convert:
                results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                           for (x, y) in data]
            else:
                results = [(np.argmax(self.feedforward(x)), y)
                           for (x, y) in data]
            return sum(int(x == y) for (x, y) in results)
        
        def total_cost(self, data, lmbda, convert=False):
            cost = 0.0
            for x, y in data:
                a = self.feedforward(x)
                if convert: y = vectorized_result(y)
                cost += self.cost.func(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(
                np.linalg.norm(w)**2 for w in self.weights)
            return cost
                
        def save(self, filename):
            data = {"sizes": self.sizes,
                    "weights": [w.tolist() for w in self.weights],
                    "biases": [b.tolist() for b in self.biases],
                    "cost": str(self.cost.__name__)}
            with open(filename, "w") as f: json.dump(data, f)

def load(filename):
    #load a neural network
    with open(filename, "r") as f: data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

        
    
    
    