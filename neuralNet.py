import numpy as np
import dill

class neuralNet:

    def __init__ (self, numLayers, numNodes, activationFunc, costFunc):
        self.numLayers = numLayers
        self.numNodes = numNodes
        self.layers = []
        self.costFunc = costFunc

        if not numLayers == len(numNodes):
            raise ValueError("Number of layers must be equal to number of node counts")

        for i in range(numLayers):
            if i != numLayers-1:
                iLayer = layer(numNodes[i], numNodes[i+1], activationFunc[i])
            else:
                iLayer = layer(numNodes[i], 0, activationFunc[i])
            self.layers.append(iLayer)

    def compareTrainingSet (self, batchSize, data, flags):
        self.batchSize = batchSize
        if not len(data) % self.batchSize == 0:
            raise ValueError("Batch size must be multiple of data size")
        if not len(data) == len(flags):
            raise ValueError("Number of inputs must match number of flags")
        for i in range(len(data)):
            if not len(data[i]) == self.numNodes[0]:
                raise ValueError("Length of each input must match number of input nodes")
            if not len(flags[i]) == self.numNodes[-1]:
                raise ValueError("Length of each flag must match number of output nodes")

    def trainer (self, batchSize, data, flags, numPasses, learningRate, dataFile):
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.compareTrainingSet(self.batchSize, data, flags)
        for j in range(numPasses):
            i = 0
            print("== ITERATIONS: ", j+1, "/", numPasses, " ==")
            while i+batchSize != len(data):
                print("Training with ", i+batchSize+1, "/", len(data), end="\r")
                self.error = 0
                self.passForward(data[i:i+batchSize])
                self.errorCalc(flags[i:i+batchSize])
                self.passBackward(flags[i:i+batchSize])
                i += batchSize
            self.error /= batchSize
            print("\nError: ", self.error)
        print("Saving...")
        dill.dump_session(dataFile)

    def passForward (self, data):
        self.layers[0].activations = data
        for i in range(self.numLayers-1):
            temp = np.add(np.matmul(self.layers[i].activations, self.layers[i].layerWeights), self.layers[i].layerBias)
            if self.layers[i+1].activationFunc == "sigmoid":
                self.layers[i+1].activations = self.sigmoid(temp)
            elif self.layers[i+1].activationFunc == "softMax":
                self.layers[i+1].activations = self.softMax(temp)
            elif self.layers[i+1].activationFunc == "rectify":
                self.layers[i+1].activations = self.rectify(temp)
            else:
                self.layers[i+1].activations = temp

    def rectify(self, layer):
        layer[layer < 0] = 0
        return layer

    def softMax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp/np.sum(exp, axis=1, keepdims=True)
        else:
            return exp/np.sum(exp, keepdims=True)

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(np.negative(layer))))


    def errorCalc(self, flags):
        if len(flags[0]) != self.layers[self.numLayers-1].numNodesCurLayer:
            print ("Error: Flag is not of the same shape as output layer.")
            print("Flag: ", len(flags), " : ", len(flags[0]))
            print("Output: ", len(self.layers[self.numLayers-1].activations), " : ", len(self.layers[self.numLayers-1].activations[0]))
            return

        if self.costFunc == "meanSquareMethod":
            self.error += np.mean(np.divide(np.square(np.subtract(flags, self.layers[self.numLayers-1].activations)), 2))
        elif self.costFunc == "crossEntropyMethod":
            self.error += np.negative(np.sum(np.multiply(flags, np.log(self.layers[self.numLayers-1].activations))))

    def passBackward(self, flags):
        # only if self.costFunc == "crossEntropyMethod" and self.layers[self.numLayers-1].activationFunc == "softMax":
        targets = flags
        i = self.numLayers-1
        y = self.layers[i].activations
        deltaB = np.multiply(y, np.multiply(1-y, targets-y))
        deltaW = np.matmul(np.asarray(self.layers[i-1].activations).T, deltaB)
        weightsNew = self.layers[i-1].layerWeights - self.learningRate * deltaW
        biasNew = self.layers[i-1].layerBias - self.learningRate * deltaB
        for i in range(i-1, 0, -1):
            y = self.layers[i].activations
            deltaB = np.multiply(y, np.multiply(1-y, np.sum(np.multiply(biasNew, self.layers[i].layerBias)).T))
            deltaW = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, np.sum(np.multiply(weightsNew, self.layers[i].layerWeights),axis=1).T)))
            self.layers[i].layerWeights = weightsNew
            self.layers[i].layerBias = biasNew
            weightsNew = self.layers[i-1].layerWeights - self.learningRate * deltaW
            biasNew = self.layers[i-1].layerBias - self.learningRate * deltaB
        self.layers[0].layerWeights = weightsNew
        self.layers[0].layerBias = biasNew


    def predict(self, dataFile, input):
        dill.load_session(dataFile)
        self.batchSize = 1
        self.passForward(input)
        a = self.layers[self.numLayers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        return a

    def checkAccuracy(self, dataFile, data, flags):
        dill.load_session(dataFile)
        self.batchSize = len(data)
        self.passForward(data)
        a = self.layers[self.numLayers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        total=0
        correct=0
        for i in range(len(a)):
            total += 1
            if np.equal(a[i], flags[i]).all():
                correct += 1
        print("Accuracy: ", correct*100/total)



    def loadModel(self, dataFile):
        dill.load_session(dataFile)


class layer:
    def __init__(self, numNodesCurLayer, numNodesNextLayer, activationFunc):
        self.numNodesCurLayer = numNodesNextLayer
        self.activationFunc = activationFunc
        self.activations = np.zeros([numNodesCurLayer,1])
        if numNodesNextLayer != 0:
            self.layerWeights = np.random.normal(0, 0.001, size=(numNodesCurLayer, numNodesNextLayer))
            self.layerBias = np.random.normal(0, 0.001, size=(1, numNodesNextLayer))
        else:
            self.layerWeights = None
            self.layerBias = None
