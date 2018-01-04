import numpy as np
import pandas as pd

trainInput = pd.read_csv('train_X_dog_cat.csv',header=None).as_matrix()
trainOutput = pd.read_csv('train_y_dog_cat.csv',header=None).as_matrix()
#print("shape of trainOutput", trainOutput.shape)

dataPoints = trainInput.shape[0]
dimensions = trainInput.shape[1]
maxItr = 50
lamdaV = 1000


def calculateAlphaVector(trainInput, weight,trainOutput):
    XWeight = np.dot(trainInput,weight)
    return 1/(1+np.exp(trainOutput*XWeight))
    


def calHessAndGradient(trainInput, trainOutput,weight,lVal, regCoeffVector,iterationNumber):
    dimensions = trainInput.shape[1]
    alphaVector = calculateAlphaVector(trainInput,weight,trainOutput)
    #print("alphaVector", alphaVector)
    
    #Calculating Gradient
    alphaOutput = regCoeffVector*alphaVector * trainOutput #(1953, 1)
    gradient = (-1)*np.dot(np.transpose(trainInput), alphaOutput)
    ## Would need to separate above code into positive and negative classes
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

finalWeight = logisticRegression(trainInput,trainOutput,lamdaV)
#print("Final Weight Vector", finalWeight)

##TESTING STEP
testInput = pd.read_csv('test_X_dog_cat.csv',header=None).as_matrix()
testOutput = pd.read_csv('test_y_dog_cat.csv',header=None).as_matrix()
#print("testInput shape", testInput.shape)

mistakeCount = 0
for i in range(testInput.shape[0]):
    if np.dot(np.transpose(finalWeight),testInput[i]) >=0 :
        if(testOutput[i] != 1):
            mistakeCount+=1
    else:
        if(testOutput[i] != -1):
            mistakeCount+=1
    
print("MistakeCount on test set:", mistakeCount, mistakeCount/testInput.shape[0])


