import numpy as np
import pandas as pd
import timeit
from sklearn import preprocessing

start = timeit.default_timer()
trainInput = pd.read_csv('train_X_dog_cat.csv',header=None).as_matrix()
trainOutput = pd.read_csv('train_y_dog_cat.csv',header=None).as_matrix()
print("shape of trainOutput", trainOutput.shape)

trainInput = preprocessing.scale(trainInput)
## Standardizing train Input :: x-mean/std dev



dataPoints = trainInput.shape[0]
dimensions = trainInput.shape[1]
maxItr = 50
lamdaV = 2000
sigma = 1000
polyPower = 5

#Following square Matrix will be used over and over again in evaluation of GRAM Matrices of various kernels
XXT = np.dot(trainInput,np.transpose(trainInput))
print("XXT shape", XXT.shape) #1953*1953

#print("GAUSSIAN SHAPE", calculateRegressionCoefficientVector(trainOutput).shape)
diagnolVector = np.diagonal(XXT) # extracting diagnol elemnts from XXT as it has x1 norm, x2 norm ...
reshapeDiagVector = diagnolVector.reshape(1,diagnolVector.shape[0]) # (1,n)


def calculateHessianAndGradient(gramMatrix, coeffVector, trainOutput,lamdaV, alphaVector):
    dataPoints = gramMatrix.shape[0]
    betaVector = evaluateBetaVector(gramMatrix, alphaVector, trainOutput)
    
    coeffBeta = coeffVector*betaVector
    #print("CoeffBeta shape", coeffBeta)
    #CalculateGradient
    gradient = (-1.0)*coeffBeta*trainOutput
    gradient = np.dot(gramMatrix,gradient)
    #print("gradient shape", gradient)
    zz = (2.0)*lamdaV*(np.dot(gramMatrix,alphaVector))
    #print("zz shape", zz)
    gradient = np.add(gradient,zz)
    #gradient = np.dot(gramMatrix,gradient)
    
    #Hessian
    identityMatrix = np.identity(dataPoints)
    changedBetaVector = 1-betaVector
    hessian = coeffBeta*changedBetaVector*gramMatrix
    hessian = np.add(np.transpose(hessian), 2*lamdaV*identityMatrix)
    hessian = np.dot(hessian,gramMatrix)
    return (hessian,gradient)
    

def calculateRegressionCoefficientVector(trainOutput):
    positiveCounts = (trainOutput == 1).sum()
    negativeCounts = trainOutput.shape[0]-positiveCounts
    #print("Positive count, Negative count", positiveCounts, negativeCounts)
    coeffVector = np.full((trainOutput.shape[0],1),0.0)
    for i in range(trainOutput.shape[0]):
        yVal = trainOutput[i]
        coeffVector[i] = 1/(positiveCounts*((yVal+1)/2) + negativeCounts*((1-yVal)/2))
    #print("Coeff vector shape", coeffVector)
    return coeffVector # This vector of dimension n*1


def linearGramMatrix(XXT):
    return XXT

def polynomialGramMatrix(XXT):
    #matrixCopy = np.copy(XXT)
    return (1+XXT)**polyPower

def gaussianGramMatrix(XXT, sigma,reshapeDiagVector):
    #matrixCopy = np.copy(XXT)
    dimOfMatrix = reshapeDiagVector.shape[1]
    tiledMatrix = np.tile(reshapeDiagVector,(dimOfMatrix,1))
    transposedMatrix = np.transpose(tiledMatrix)
    transposedMatrix = reshapeDiagVector + transposedMatrix # broadcasting
    sub = np.subtract(transposedMatrix,2*XXT)
    outputGramMatrix = (-1.0/sigma)*(sub)
    outputGramMatrix = np.exp(outputGramMatrix)
    print(outputGramMatrix[0])
    return outputGramMatrix# n*n 1953*1953
    

def evaluateBetaVector(gramMatrix, alphaVector, outputVector):
    gAlpha = np.dot(gramMatrix, alphaVector)
    return   1.0/(1+ np.exp(outputVector*gAlpha))
    

def logisticRegression(trainInput, trainOutput,lVal, gramMatrix):
    alphaVector = np.full((dataPoints,1),0.0)
    finalAlpha = alphaVector
    #finalGrad = np.full((dimensions,1),0) # (3072,1)
    #finalHess = np.full((dimensions,dimensions),0) #(3072,3072)
    
    regCoeffVector = calculateRegressionCoefficientVector(trainOutput)
    for k in range(maxItr):
        #print("Iteration Number :::: ", k+1)
        oldAlpha = finalAlpha
        (hess, grad) = calculateHessianAndGradient(gramMatrix,regCoeffVector,trainOutput,lVal, alphaVector)
        alphaDash = np.linalg.solve(hess,grad)
        finalAlpha = np.subtract(oldAlpha,alphaDash)
        #finalGrad = grad
        #finalHess = hess
        #print("grad", finalGrad)
        #print("hess", finalHess)
        diffWeightVectors = np.subtract(oldAlpha,finalAlpha)
        if(abs(np.linalg.norm(diffWeightVectors)) < 10**-4):
            break;
            
    #print("finalWeigt magnitude and dimension", np.dot(np.transpose(finalWeight), finalWeight), finalWeight)
    return finalAlpha

finalAlphaLinear = logisticRegression(trainInput, trainOutput,lamdaV,linearGramMatrix(XXT))
finalAlphaPolynomial = logisticRegression(trainInput, trainOutput,lamdaV,polynomialGramMatrix(XXT))
finalAlphaGaussian = logisticRegression(trainInput, trainOutput,lamdaV,gaussianGramMatrix(XXT,sigma,reshapeDiagVector))

print("finalAlphaLinear", finalAlphaLinear.shape, np.dot(np.transpose(finalAlphaLinear),finalAlphaLinear))
print("finalAlphaPolynomial", finalAlphaPolynomial.shape, np.dot(np.transpose(finalAlphaPolynomial),finalAlphaPolynomial))
print("finalAlphaGaussian", finalAlphaGaussian.shape, np.dot(np.transpose(finalAlphaGaussian),finalAlphaGaussian))




# Testing
testInput = pd.read_csv('test_X_dog_cat.csv',header=None).as_matrix()
testOutput = pd.read_csv('test_y_dog_cat.csv',header=None).as_matrix()

testInput = preprocessing.scale (testInput)


print("testInput shape", testInput.shape)
print("testOutput shape", testOutput.shape)
reshapedTestOutput = np.reshape(testOutput,(1,testOutput.shape[0]))
print("reshapedTestOutput shape", reshapedTestOutput.shape)
#Reused over and over again in different kernels
XTrainXTestTr = np.dot(trainInput,np.transpose(testInput))

linearKernelPredictions = np.dot(np.transpose(finalAlphaLinear),XTrainXTestTr) # 1*testPints vector
polyKernelPredictions = np.dot(np.transpose(finalAlphaPolynomial),(1+XTrainXTestTr)**polyPower) # 1*testPints vector

#For Gaussian
dimOfMatrix = testOutput.shape[0]
tiledMatrix = np.tile(reshapeDiagVector,(dimOfMatrix,1))
transposedMatrix = np.transpose(tiledMatrix)## n*t matrix

l2NormTestData = (np.square(testInput).sum(axis=1)).reshape(1,testOutput.shape[0])
transposedMatrix = l2NormTestData + transposedMatrix # broadcasting
sub = np.subtract(transposedMatrix,2*XTrainXTestTr)
testGaussMatrix = (-1.0/sigma)*(sub)
testGaussMatrix = np.exp(testGaussMatrix)  # n*t 
gaussKernelPredictions = np.dot(np.transpose(finalAlphaGaussian),testGaussMatrix)

#Gaussian ends

linearKernelPredictions[linearKernelPredictions>=0] = 1
linearKernelPredictions[linearKernelPredictions<0] = -1

polyKernelPredictions[polyKernelPredictions>=0] = 1
polyKernelPredictions[polyKernelPredictions<0] = -1

gaussKernelPredictions[gaussKernelPredictions>=0] = 1
gaussKernelPredictions[gaussKernelPredictions<0] = -1

misClassLinear = ((np.subtract(linearKernelPredictions,reshapedTestOutput))!=0).sum()
misClassPoly = ((np.subtract(polyKernelPredictions,reshapedTestOutput))!=0).sum()
misClassGauss = ((np.subtract(gaussKernelPredictions,reshapedTestOutput))!=0).sum()

print("Linear Misclassification", misClassLinear, misClassLinear/testOutput.shape[0])
print("Poly Misclassification", misClassPoly, misClassPoly/testOutput.shape[0])
print("Gaussian Misclassification", misClassGauss, misClassGauss/testOutput.shape[0])
## Testing

stop = timeit.default_timer()
print ("TOTAL TIME", stop - start)







 


