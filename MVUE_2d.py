import numpy as np
import math
import random
from sympy import *
import time
import pickle
import scipy.signal
import numpy

# Updated --> changed COTR and MatrixFunc as in V3

# For functions with doc-string, it has been updated for 2d case.

# Assume that the two dimensions are the same.
def getShiftBoxes_2d(lengthOfChip, lengthOfROI):
    if lengthOfROI>=lengthOfChip:
        return ('length error')
    else:
        arrayOfAll=np.zeros(((lengthOfChip-lengthOfROI+1)**2,lengthOfChip,lengthOfChip),float);
        count=0;
        for i in range(lengthOfChip-lengthOfROI+1):
            for j in range(lengthOfChip-lengthOfROI+1):
                arrayOfAll[count,i:i+lengthOfROI,j:j+lengthOfROI]=1
                count+=1;
        return arrayOfAll

def coefficientSize_V2(lengthOfChip, logBinUpTo):
    """Return total number of coefficients (for 1d case)."""
    Size=0;
    for i in range(logBinUpTo+1):
        Size+= lengthOfChip // 2**i 
    return Size

# Note the sequence of basis, logIndexX is bigger than logIndexY, so we have the same logIndexX first, 
# and the logIndexY is changed.
def generateBasis_2d(lengthOfChip, logBinUpTo):
    """Generate Basis for 2d chip."""
    basisList=np.zeros((coefficientSize_V2(lengthOfChip, logBinUpTo)**2,lengthOfChip,lengthOfChip));
    count=0;
    for logIndexX in range(0,logBinUpTo+1):
        for logIndexY in range(0,logBinUpTo+1):
            for binIndexX in range(0,lengthOfChip//(2**logIndexX)):
                for binIndexY in range(0,lengthOfChip//(2**logIndexY)):
                    tempBasis=np.zeros((lengthOfChip,lengthOfChip));
                    tempBasis[binIndexX*2**logIndexX:(binIndexX+1)*2**logIndexX, \
                              binIndexY*2**logIndexY:(binIndexY+1)*2**logIndexY]=1
                    basisList[count, :, :]=tempBasis
                    count+=1;
    return basisList

def generateGT_2d(numOfSamples, chipSize):
    """Generate GT for 2d samples."""
    GT = np.random.rand(chipSize, chipSize, numOfSamples);
    GT = (GT/np.sum(np.sum(GT,0),0)) *100 # Normalize each sample to 100
    return GT # Shape of (chipSize, chipSize, numOfSamples)

def convertCoordinatesToTargetBox(startingCoordinates, endingCoordinates, lengthOfChip):
    """Give tuples as coordinates, return the targetBox in matrix format."""
    targetBox=np.zeros((lengthOfChip, lengthOfChip));
    targetBox[startingCoordinates[0]:endingCoordinates[0],startingCoordinates[1]:endingCoordinates[1]]=1;
    return targetBox

def generateEstimatorList_2d(targetBox, basisList, lengthOfChip, logBinUpTo):
    """Generate estimatorList for 2d."""
    estimatorList=np.zeros((coefficientSize_V2(lengthOfChip, logBinUpTo)**2-lengthOfChip**2+1,
                            coefficientSize_V2(lengthOfChip, logBinUpTo)**2));
    # Single pixels first
    estimatorList[0,:lengthOfChip**2]=targetBox.flatten();
    differenceMatrix=targetBox-basisList
    count=0;
    for i in range(lengthOfChip**2, basisList.shape[0]):
        count+=1;
        tempVector=np.zeros(basisList.shape[0]);
        tempVector[i]=1;
        tempVector+=np.concatenate((differenceMatrix[i,:].flatten(), np.zeros(basisList.shape[0]-lengthOfChip**2)))
        estimatorList[count,:]=tempVector
    # for index in range(0,count+1): # For testing
    #     print(estimatorList[index][:lengthOfChip**2].reshape(4,4))
    #     print(estimatorList[index][lengthOfChip**2:])
    return estimatorList

def binning2D(GT, binningSize, exposureTime=1):
    """binningSize expects a tuple."""
    numOfSamples=GT.shape[2]
    return scipy.signal.convolve(GT,np.ones((binningSize[0],binningSize[1],1)),mode='valid')[::binningSize[0], ::binningSize[1],:]*exposureTime

def addNoise(binnedGT, readNoiseSD):
    """Just adding Poisson and Gaussian Noises to the binned GT."""
    return np.random.poisson(binnedGT) + np.random.normal(loc = 0, scale=readNoiseSD, size=binnedGT.shape)

def findMVUE(fileName, targetBox=np.ones((4,4)), lengthOfChip=4, logBinUpTo=2):
    """Return a dictionary storing matrices for MVUE."""

    if logBinUpTo==None:
        logBinUpTo=int(np.log2(lengthOfChip));

    basisList=generateBasis_2d(lengthOfChip, logBinUpTo);
    estimatorList=generateEstimatorList_2d(targetBox, basisList, lengthOfChip, logBinUpTo)

    # Declare a sympy array, storing all the weights
    weightArray=symarray('w',len(estimatorList))
    weightArray[-1]=1-(np.sum(weightArray)-weightArray[-1]) ### added 

    # NOTE THE ORDER OF ESTIMATORS - IT FOLLOWS ORDER OF BASIS, SO LOGBININDEX Y CHANGES FIRST
    allEstimator=np.zeros(estimatorList[0].shape)
    for i in range(len(estimatorList)):
        allEstimator=allEstimator+estimatorList[i]*weightArray[i]
        
    # No covariance between different estimators, since they are independent measurements
    varianceCoefArray=allEstimator*allEstimator

    # A matrix storing the langrangian differentiated w.r.t. each of the w_s
    diffMatrix=np.empty((len(allEstimator),len(weightArray)-1),dtype=object); 
    for i in range(len(allEstimator)):
        for j in range(len(weightArray)-1):
            diffMatrix[i,j]=diff(varianceCoefArray[i],weightArray[j])

    varianceArray=symarray('V',(len(allEstimator)))
    finalSolve=np.dot(diffMatrix.T,varianceArray)

    # t=time.time();
    finalSolveWeightCoeffMatrix=np.empty((len(finalSolve),len(weightArray)-1),dtype=object);
    for i in range(len(finalSolve)):
        for j in range(len(weightArray)-1):
            finalSolveWeightCoeffMatrix[i,j]=diff(finalSolve[i],weightArray[j])
            # finalSolveWeightCoeffMatrix[i,j]=Poly(finalSolve[i],weightArray[j]).coeff_monomial(weightArray[j])

    constantOnTheRight=-finalSolve;
    subsDict={weight: 0 for weight in weightArray}
    for w1 in range(len(weightArray)-1):
        constantOnTheRight[w1]=constantOnTheRight[w1].subs(subsDict)

    # constantOnTheRight=np.empty(len(finalSolve),dtype=object);
    # for i in range(len(finalSolve)):
    #     constantOnTheRight[i]=simplify(finalSolve[i]-np.dot(finalSolveWeightCoeffMatrix[i,:], weightArray[:-1]))*-1
    finalSolveWeightCoeffMatrixSympy=Matrix(finalSolveWeightCoeffMatrix);
    # matrixFunc=lambdify(varianceArray, finalSolveWeightCoeffMatrixSympy, numpy);
    # COTRFunc=lambdify(varianceArray, Matrix(constantOnTheRight), numpy);
    # print(time.time()-t)

    saveDict={'finalSolveWeightCoeffMatrixSympy':finalSolveWeightCoeffMatrixSympy, 'constantOnTheRight':Matrix(constantOnTheRight), 'varianceArray':varianceArray,'basisList':basisList,'weightArray':weightArray,'estimatorList':estimatorList}

    with open(fileName, 'wb') as handle:
        pickle.dump(saveDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return saveDict

    