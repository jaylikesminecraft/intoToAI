import math
import random
import matplotlib.pyplot as plt

#Forward propogation
# if layer variable is 0 this returns the predicted output of the network
# if layer variable is 1 it returns the outputs of the hidden layer
# not effecient when calling a previous layer because each neuron output is recomputed instead of saved
def network(x: list, networkWeights: list, layer: int):
    inputs = x
    hiddenOutputs = []
    hiddenOutputs.append(inputs)
    # i is the layer
    for i in range(layer):
        layerOutput = []
        # j in the neuron's weights in that layer
        for j in range(len(networkWeights[i])):
            #select the neuron to work on
            weights = networkWeights[i][j]
            #save the output of that neuron
            layerOutput.append(sigmoidEstimation(inputs, weights))
        #in the next layer use the outputs from this layer as the inputs
        hiddenOutputs.append(layerOutput)
        inputs = layerOutput

    return inputs, hiddenOutputs

def D_sigmoid(z):
    return z * (1 - z)

def networkBackPropogation(x: list, y: list, networkWeights: list, iterations, step):

    neuronError = []
    mse = []
    numLayers = len(networkWeights)
    finalLayer = numLayers - 1

    #build an array structure to store the error for each neuron
    for layers in range(len(networkWeights)):
        layer = []
        for neuron in range(len(networkWeights[layers])):
            layer.append(0)
        neuronError.append(layer)
            
    for i in range(iterations):
        #for every piece of data
        errorAtInteration = []
        for j in range(len(y)):
            
            for numberOfOutputNeurons in range(len(networkWeights[finalLayer])):
                
                #forward propogation
                output, hiddenOutputs = network(x[j], networkWeights, len(networkWeights))
                error = y[j][numberOfOutputNeurons] - output[numberOfOutputNeurons]
                #save error to graph
                errorAtInteration.append(error ** 2)
                #derivative of error
                dError = error * D_sigmoid(output[numberOfOutputNeurons])

                #get the values from the layer just before
                hiddenOutput = hiddenOutputs[finalLayer]

                #start the backpropogation from the output layer
                for weight in range(len(networkWeights[finalLayer][numberOfOutputNeurons])):
                    if (weight == 0):
                        networkWeights[finalLayer][numberOfOutputNeurons][weight] += step * dError
                    else:
                        neuronError[finalLayer - 1][weight - 1] = (dError * D_sigmoid(hiddenOutput[weight-1]) * networkWeights[finalLayer][numberOfOutputNeurons][weight])
                        networkWeights[finalLayer][numberOfOutputNeurons][weight] += step * hiddenOutput[weight - 1] * dError

                #start backpropogating through each layer
                for currentLayer in reversed(range(finalLayer)):
                    #print("hiddenlayer")
                    hiddenOutput = hiddenOutputs[currentLayer]

                    for neuron in range(len(networkWeights[currentLayer])):
                        for weight in range(len(networkWeights[currentLayer][neuron])):
                            if (weight == 0):
                                #adjust the bias
                                networkWeights[currentLayer][neuron][weight] += step * neuronError[currentLayer][neuron]
                            else:
                                #adjust the weights
                                networkWeights[currentLayer][neuron][weight] += step * neuronError[currentLayer][neuron] * hiddenOutput[weight - 1]

                        for weight in range(len(networkWeights[currentLayer][neuron])):
                            if not (weight == 0):
                                neuronError[currentLayer][neuron] = (dError * D_sigmoid(hiddenOutput[weight - 1]) * networkWeights[currentLayer][neuron][weight])
            mse.append(errorAtInteration)
            
        if (max(mse[i]) < 0.01):
            print("Stopping at iteration", i)
            break
        
    return networkWeights, mse

def sigmoidEstimation(x: list, weights: list):
    sum = weights[0] # bias
    for i in range(len(x)):
        sum += weights[i + 1] * x[i]
    return sigmoid(sum)

def sigmoid(z):
    a = 1 / (1 + math.exp((-1) * z))
    return a

def printMatrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])

def letterA():
    A = [0, 0, 1, 0, 0,
         0, 1, 0, 1, 0,
         0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1]
    return A 

def letterB():
    B = [1, 1, 1, 0, 0,
         1, 0, 0, 1, 0, 
         1, 1, 1, 0, 0,
         1, 0, 0, 1, 0,
         1, 1, 1, 0, 0]
    return B

def letterC():
    C = [0, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 0,
         1, 0, 0, 0, 1,
         0, 1, 1, 1, 0]
    return C

def letterD():
    D = [1, 1, 1, 1, 0,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 0, 0, 0, 1,
         1, 1, 1, 1, 0]
    return D

def letterE():
    E = [1, 1, 1, 1, 1,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1,
         1, 0, 0, 0, 0,
         1, 1, 1, 1, 1]
    return E

def buildNetwork(neuronsPerLayer, numberOfXInputs):

    networkWeights = []
    weightsPerNeuron = numberOfXInputs

    for layers in range(len(neuronsPerLayer)):
        networkWeights.append([])
        
        for neurons in range((neuronsPerLayer[layers])):
            neuron = [1] #Default bias
            
            for weights in range((weightsPerNeuron)):
                neuron.append(random.uniform(-1, 1))
                
            networkWeights[layers].append(neuron)
            
        weightsPerNeuron = neuronsPerLayer[layers]
    
    return networkWeights

def printResuts(networkWeights):
    for layer in range(len(networkWeights)):
        for neuron in range(len(networkWeights[layer])):
            print("neuron[" + str(layer) + "][" + str(neuron) + "] = ", networkWeights[layer][neuron])

    for i in range(len(x)):
        output = network(x[i], networkWeights, len(networkWeights))
        for j in range(len(output[0])):
            output[0][j] = round(output[0][j],2)
        print(str(x[i]) + " = " + str(output[0]))
    
#Data set
x = [letterA(), letterB(), letterC(), letterD(), letterE()]
y = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

networkWeights = buildNetwork([3, len(x)], len(x[0]))

#Train the network
networkWeights, mse = networkBackPropogation(x, y, networkWeights, 20000, 3)
printResuts(networkWeights)

plt.plot(mse)
plt.title("Iteration Number vs Error")
plt.xlabel("Iteration Number")
plt.ylabel("Error")
plt.show()
