import numpy as np
from collections import OrderedDict
nameInputDict = OrderedDict()
nameInputDict["CIFAR_10_TRAIN_1"] = '/Users/hardiksahi/Desktop/Assignment 3/cifar-10-batches-py/data_batch_1'
#nameInputDict["CIFAR_10_TRAIN_2"] = '/Users/hardiksahi/Desktop/Assignment 3/cifar-10-batches-py/data_batch_2'
#nameInputDict["CIFAR_10_TRAIN_3"] = '/Users/hardiksahi/Desktop/Assignment 3/cifar-10-batches-py/data_batch_3'
#nameInputDict["CIFAR_10_TRAIN_4"] = '/Users/hardiksahi/Desktop/Assignment 3/cifar-10-batches-py/data_batch_4'
#nameInputDict["CIFAR_10_TRAIN_5"] = '/Users/hardiksahi/Desktop/Assignment 3/cifar-10-batches-py/data_batch_5'

maxItr = 50
lamdaV = 20
def calculateAlphaVector(trainInput, weight,trainOutput):
    XWeight = np.dot(trainInput,weight)
    return 1/(1+np.exp(trainOutput*XWeight))

def calHessAndGradient(trainInput, trainOutput,weight,lVal, regCoeffVector,iterationNumber):
    dimensions = trainInput.shape[1]
    alphaVector = calculateAlphaVector(trainInput,weight,trainOutput)
    
    #Calculating Gradient
    alphaOutput = regCoeffVector*alphaVector * trainOutput #(1953, 1)
    gradient = (-1)*np.dot(np.transpose(trainInput), alphaOutput)
    gradient = np.add(gradient, 2*lVal*weight) #(3072, 1)
    
    #Calculating Hessian
    changedAlphaVector = 1-alphaVector 
    changedAlphaVector = regCoeffVector*changedAlphaVector*alphaVector    
    hessian = changedAlphaVector * trainInput
    hessian = np.dot(np.transpose(hessian),trainInput)
    
    identityMatrix = np.identity(dimensions)
    hessian = np.add(hessian,2*lVal*identityMatrix)    
    return (hessian, gradient)

def calculateRegressionCoefficientVector(trainOutput):
    positiveCounts = (trainOutput == 1).sum()
    negativeCounts = trainOutput.shape[0]-positiveCounts
    #print("Positive count, Negative count", positiveCounts, negativeCounts)
    coeffVector = np.full((trainOutput.shape[0],1),0.0)
    for i in range(trainOutput.shape[0]):
        yVal = trainOutput[i]
        coeffVector[i] = 1/(positiveCounts*((yVal+1)/2) + negativeCounts*((1-yVal)/2))
    #print("Coeff vector shape", coeffVector)
    return coeffVector

def logisticRegression(trainInput, trainOutput,lVal):
    weightVector = np.full((dimensions,1),0)
    finalWeight = weightVector
    regCoeffVector = calculateRegressionCoefficientVector(trainOutput)
       
    for k in range(maxItr):
        #print("Iteration Number :::: ", k+1)
        oldWeight = finalWeight
        (hess, grad) = calHessAndGradient(trainInput,trainOutput,oldWeight,lamdaV, regCoeffVector,k+1)
        weightDash = np.linalg.solve(hess,grad)
        finalWeight = np.subtract(oldWeight,weightDash)
        diffWeightVectors = np.subtract(oldWeight,finalWeight)
        if(abs(np.linalg.norm(diffWeightVectors)) < 10**-4):
            break;
            
    return finalWeight

def setCorrespondingClassPlusOne(posClassN, output):
    for k in range(output.shape[0]):
        if(output[k] == posClassN):
            output[k] = 1
        else:
            output[k] = -1
    return output

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def createInputAndOutputData():
    for key in nameInputDict:
        #print(key)
        dictInput = unpickle(nameInputDict[key])
        for k in dictInput:
            keyStr = k.decode("utf-8")
            if(keyStr == 'data'):
                if key == "CIFAR_10_TRAIN_1":
                    trainInput = dictInput[k]
                else:
                    trainInput = np.concatenate((trainInput,dictInput[k]), axis = 0)
                
            
            if(keyStr == 'labels'):
                labelList = dictInput[k]
                listAsArray = np.asarray(labelList).reshape(len(labelList),1)
                if key == "CIFAR_10_TRAIN_1":
                    trainOutput = listAsArray
                else:
                    trainOutput = np.concatenate((trainOutput,listAsArray), axis = 0)
                     
    return(trainInput,trainOutput)

(trainInput, trainOutput) = createInputAndOutputData()
#print("train input and output shape after creation",trainInput.shape, trainOutput.shape)

dataPoints = trainInput.shape[0]
dimensions = trainInput.shape[1]

wtVectorMatrix = None
for classN in range(0,10):
    positiveClassNumber = classN
    trainOutputCopy = np.copy(trainOutput)
    updateOutputVec = setCorrespondingClassPlusOne(positiveClassNumber,trainOutputCopy)
    #print("Updated output shape", updateOutputVec.shape)
    if classN == 0:
        wtVectorMatrix= logisticRegression(trainInput, updateOutputVec,lamdaV)
    else:
        wtVectorForClass = logisticRegression(trainInput, updateOutputVec, lamdaV)
        wtVectorMatrix = np.concatenate((wtVectorMatrix,wtVectorForClass), axis = 1)

print("Final weight Matrix::", wtVectorMatrix.shape)

## TESTING 
CIFAR_10_TEST = '/Users/hardiksahi/Desktop/Assignment 3/cifar-10-batches-py/test_batch'
testDict = unpickle(CIFAR_10_TEST)

testInput = None
testOutput = None

for z in testDict:
    keyStr = z.decode("utf-8")
    if(keyStr == 'data'):
        testInput = testDict[z]
        
    if(keyStr == 'labels'):
        labelList = testDict[z]
        testOutput = np.asarray(labelList).reshape(len(labelList),1)

dimTest = testInput.shape[1]
misClassificationTest = 0       
for i in range(testInput.shape[0]):
    reshapeInput = testInput[i].reshape(dimTest,1) # column vector
    weightInput = np.dot(np.transpose(reshapeInput), wtVectorMatrix)
    classPredicted = np.argmax(weightInput)
    
    if classPredicted != testOutput[i]:
        misClassificationTest+=1

print("Misclassified points in test data is %d and %f " % (misClassificationTest, misClassificationTest/testOutput.shape[0]))
        

                

                   

                
                
                
    
    