#Python derivative from URL: http://inkdrop.net/dave/docs/neural-net-tutorial.cpp

import math
import random
import sys
import numpy

class Neuron:
    learning_rate = 0.2
    momentum_rate = 0.2
    weight = None
    output = None
    feedforward_weights = None
    feedforward_dweights = None
    neuron_num = None
    gradient = None
    
    def __init__(self, outputs, neuron_num):
        self.feedforward_weights = list()
        self.feedforward_dweights = list()
        for i in range(0, outputs):
            self.feedforward_weights.append(random.uniform(0,1))
            self.feedforward_dweights.append(0)
        self.neuron_num = neuron_num
        
    def feedforward(self, lower_layer):
        value = 0.0
        for i in range(0, len(lower_layer)):
            value += (lower_layer[i].output * lower_layer[i].feedforward_weights[self.neuron_num])
        self.output = self.calc_activation_function_val(value)
        
    def calc_activation_function_val(self, x):
        function = 1/(1+numpy.exp(-1*x))
        return function
        
    def calc_activation_function_dval(self, x):
        function = math.exp(-x)/math.pow(1+numpy.exp(-x), 2)
        return function
    
    #Source Code Reference 2
    def compute_output_grad(self, target):
        delta = target - self.output
        self.gradient = delta * self.calc_activation_function_dval(self.output)
    
    #Source Code Reference 3 - last hidden layer
    def compute_hidden_grad(self, upper_layer):
        contribution_error = self.get_contribution_error(upper_layer)
        self.gradient = contribution_error * self.calc_activation_function_dval(self.output)
    
    #Source Code Reference 6
    def recalc_input(self, lower_layer):
        for i in range(0, len(lower_layer)):
            lower_layer_neuron = lower_layer[i]
            last_dweight = lower_layer_neuron.feedforward_dweights[self.neuron_num]
            momentum = self.momentum_rate * last_dweight
            new_dweight = self.learning_rate * lower_layer_neuron.output * self.gradient + momentum
            lower_layer_neuron.feedforward_weights[self.neuron_num] += new_dweight
            lower_layer_neuron.feedforward_dweights[self.neuron_num] = new_dweight
        
    #Source Code Reference 1 - last hidden layer to the output neurons
    def get_contribution_error(self, upper_layer):
        contribution_error = 0
        for i in range(0,len(upper_layer)-1):
            contribution_error += self.feedforward_weights[i] * upper_layer[i].gradient
        return contribution_error

class Neural_Network:
    layers = None
    error = 0
    
    def __init__(self, num_neuron_and_layer):
        self.layers = list()
        num_layers = len(num_neuron_and_layer)
        for i in range(0,num_layers):
            self.layers.append(list())
            if(i == len(num_neuron_and_layer)-1):
                num_output = 0
            else:
                num_output = num_neuron_and_layer[i+1]
            for j in range(0,num_neuron_and_layer[i]+1):
                self.layers[len(self.layers)-1].append(Neuron(num_output,j))
            last_added_layer = self.layers[len(self.layers)-1]
            last_added_layer[len(last_added_layer)-1].output = 1.00
            
    def feedforward(self, inputs):
        for i in range(0,len(inputs)):
            self.layers[0][i].output = inputs[i]
        for j in range(1,len(self.layers)):
            lower_layer = self.layers[j-1]
            for k in range(0,len(self.layers[j])-1):
                self.layers[j][k].feedforward(lower_layer)
    
    def backpropagation(self, targets):
        output_layer = self.layers[len(self.layers)-1]
        self.error = 0.00
        
        #Code Reference 4
        for i in range(0,len(output_layer)-1):
            delta = targets[i] - output_layer[i].output
            self.error += math.pow(delta,2)
        self.error /= (len(output_layer)-1)
        #self.error = math.sqrt(self.error)
        
        for i in range(0,len(output_layer)-1):
            output_layer[i].compute_output_grad(targets[i])
        
        #First iteration: Code Reference 5
        for i in xrange(len(self.layers)-2,0,-1):
            hidden_layer = self.layers[i]
            upper_layer = self.layers[i+1]
            for j in range(0,len(hidden_layer)):
                hidden_layer[j].compute_hidden_grad(upper_layer)
            
        #Source Code Reference 6
        for i in range(len(self.layers)-1,0,-1):
            current_layer = self.layers[i]
            lower_layer = self.layers[i-1]
            for j in range(0,len(current_layer)-1):
                current_layer[j].recalc_input(lower_layer)
        
    def flush_results(self,results):
        for i in range(0,len(self.layers[len(self.layers)-1])-1):
            results.append(self.layers[len(self.layers)-1][i].output)
            
    def flush_weights(self,weights):
        for l in range(0,len(self.layers)):
            lst = list()
            for n in range(0,len(self.layers[l])-1):
                lst.append(self.layers[l][n].feedforward_weights)
            weights.append(lst)
    
def print_list(label,lst):
    sys.stdout.write(label + " ")
    for i in range(0,len(lst)):
        sys.stdout.write(str(lst[i]))
        sys.stdout.write(" ")
    sys.stdout.write("\n")

def setup_layers(lst):
    #input layers
    lst.append(2)
    #hidden layers
    lst.append(3)
    lst.append(3)
    #output layers
    lst.append(1)

def get_input(line):
    out = line.split()
    
    label = out[0]
    out.pop(0)
    
    num = list()
    for n in out:
        num.append(float(n))
    
    return [label,num]

def main(): 
    num_layers_neurons = list()
    setup_layers(num_layers_neurons)
    
    neural_net = Neural_Network(num_layers_neurons)
    
    results = list()
    
    textfile = open("training_data.txt", 'r')
    my_file = open("output_data.txt", "w")
    
    training_iterations = 1
    line = textfile.readline().strip()
    while(line != ""):
        print("Learning data # " + str(training_iterations))
        if(training_iterations%100000==0):
            my_file.write("Learning data # " + str(training_iterations))
            my_file.write("\n")
        [label,data] = get_input(line)
        print_list("Neural network input: ",data)
        if(training_iterations%100000==0):
            my_file.write("Neural network input: "+str(data))
            my_file.write("\n")
        neural_net.feedforward(data)
        
        results = list()
        neural_net.flush_results(results)
        print_list("Neural network output: ",results)
        if(training_iterations%100000==0):
            my_file.write("Neural network output: "+str(results))
            my_file.write("\n")
        
        line = textfile.readline().strip()
        [label,data] = get_input(line)
        print_list("Neural network target: ",data)
        if(training_iterations%100000==0):
            my_file.write("Neural network output: "+str(data))
            my_file.write("\n")
        
        neural_net.backpropagation(data)
        
        line = textfile.readline().strip()
        training_iterations += 1
        
        print(" ")
    textfile.close()
    my_file.close()
    
if(__name__ == "__main__"):
    main()
    print("End of processing.")





    