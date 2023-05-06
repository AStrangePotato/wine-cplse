import numpy as np
import csv
import pickle
import random
import pyperclip
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs) # already transposed
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues): #dvalues = gradients passed from layer to the right of this one
        self.dweights = np.dot(self.inputs.T, dvalues) #neuron function wrt weights is inputs (product rule)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) #neuron function wrt biases is 1
        self.dinputs = np.dot(dvalues, self.weights.T) #gets passed on

    def load_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

class ReLU_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
    def backward(self, dvalues):
        # Zero gradient where input values were negative
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftMax_Activation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #unnormalized
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #normalized

        self.output = probabilities

    def backward(self, dvalues): #shamelessly ripped off
    # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
            single_dvalues)

    #def backward(self, dvalues):    

class TanH_Activation:
    def forward(self, inputs):
        self.output = np.tanh(inputs)

class Loss:
    def calculate(self, output, y):
        losses = self.forward(output, y)
        average_loss = np.mean(losses)

        return average_loss #cost

class Loss_CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        clipped_y_pred = np.clip(y_pred, 1e-7, 1-1e-7) #prevents log by 0

        #only for sparse labelling
        correct_confidences = clipped_y_pred[range(len(y_pred)), y_true]

        neg_log = -np.log(correct_confidences)
        return neg_log

    def backward(self, y_pred, y_true):
        samples = len(y_pred)

        labels = len(y_pred[0]) #num of classes

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        #calculate and normalize gradient
        self.dinputs = -y_true / y_pred
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = SoftMax_Activation()
        self.loss = Loss_CrossEntropy()
        # Forward pass

    def forward(self, inputs, y_true):
    # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
    # Number of samples
        samples = len(dvalues)

        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
    # Update parameters
    def update_params(self, layer):
    # If layer does not contain cache arrays,
    # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

            # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
        weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
        bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates


    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

def save_params(filename="params.txt"):
    p = [layer1.weights, layer1.biases, layer2.weights, layer2.biases, layer3.weights, layer3.biases]
    with open(filename, 'wb') as f:
        pickle.dump(p, f)

def test_model():
    correct_guesses = 0

    for i in range(10000):
        layer1.forward(test_data[i])
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        test_loss = loss_function.forward(layer3.output, test_labels[i])

        prediction = np.argmax(loss_function.output, axis=1)
        correct_guesses += (prediction == test_labels[i])

    return correct_guesses/10000

def predict(X):
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)
    return list(activation3.output)

random.seed(4141627838)
#hyperparams 
#optimizer = Optimizer_SGD(learning_rate=0.5, decay=0.005, momentum=0.85)

layer1 = Layer(11,18)
layer2 = Layer(18,16)
layer3 = Layer(16, 10)

activation1 = ReLU_Activation()
activation2 = ReLU_Activation()
activation3 = SoftMax_Activation()

batchsize = 80

lr = 0.001
decay = 0.003

loss_function = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=lr, decay=decay)

training_data = np.load("training_data.npy")
training_labels = np.load("training_labels.npy")
test_data = np.load("test_data.npy")
test_labels = np.load("test_labels.npy")
print("Looaded training files")

plot_train = []
plot_test = []

size = 4000

for epoch in range(120):
    newEpoch = True
    for batch in range(0, size//batchsize, batchsize):

        batch_data = np.zeros((batchsize, 11))
        labels = np.zeros(batchsize)

        for sample in range(batchsize):
            sel = random.randint(0, size-1)
            batch_data[sample] = training_data[sel]
            labels[sample] = training_labels[sel]



        correct_guesses = 0
        for i in range(100):
            test_sel = random.randint(0, 897)
            layer1.forward(test_data[test_sel])
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)
            layer3.forward(activation2.output)
            test_loss = loss_function.forward(layer3.output, test_labels[test_sel].astype(int))

            prediction = np.argmax(loss_function.output, axis=1)
            correct_guesses += (prediction == test_labels[test_sel])

        test_acc = float(correct_guesses/100)

        #training
        labels = labels.astype(int)
        layer1.forward(batch_data)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)
        layer3.forward(activation2.output)
        loss = loss_function.forward(layer3.output, labels)

        predictions = np.argmax(loss_function.output, axis=1)
        accuracy = np.mean(predictions == labels)

        if newEpoch:
            plot_train.append(accuracy)
            plot_test.append(test_acc)
            newEpoch = False

        #backprop
        loss_function.backward(loss_function.output, labels)
        layer3.backward(loss_function.dinputs)
        activation2.backward(layer3.dinputs)
        layer2.backward(activation2.dinputs)
        activation1.backward(layer2.dinputs)
        layer1.backward(activation1.dinputs)

        optimizer.pre_update_params()
        optimizer.update_params(layer1)
        optimizer.update_params(layer2)
        optimizer.update_params(layer3)
        optimizer.post_update_params()


    print(f'epoch: {epoch}, ' +
        f'batch: {batch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'test acc: {test_acc:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'lr: {optimizer.current_learning_rate:.6f}')

x_values = np.arange(len(plot_train))

print(plot_test[-1])
plt.plot(x_values, plot_train, label='Train Accuracy', alpha=0.2)
plt.plot(x_values, plot_test, label='Test Accuracy', alpha=0.5)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f"Model Performance")
plt.legend()
plt.show()
