import numpy as np
import sympy as sp
import math
import random
from sympy import *
import time
import pickle
import scipy.signal
import numpy
import dill
import os

# V3:
# Changed Poly, coeff_monomial to diff @ MatrixFunc
# Changed COTR
# Removed the long blocks of commented out code

def getShiftBoxes(lengthOfChip, lengthOfROI):
    if lengthOfROI>=lengthOfChip:
        return ('length error')
    else:
        numOfZeros=lengthOfChip-lengthOfROI;
        v1=np.concatenate((np.zeros(numOfZeros),np.ones(lengthOfROI)))
        arrayOfAll=v1;
        for i in range(numOfZeros):
            v1=np.concatenate((v1[1:],v1[0:1]),axis=0)
            arrayOfAll=np.vstack((arrayOfAll,v1))
        return arrayOfAll


# def generateBasis(lengthOfChip, logBinUpTo):
#     basisList=np.zeros((coefficientSize(logBinUpTo),lengthOfChip));
#     count=0;
#     for logIndex in range(0,logBinUpTo+1):
#         for binIndex in range(0,lengthOfChip//(2**logIndex)):
#             tempBasis=np.zeros(lengthOfChip);
#             tempBasis[binIndex*2**logIndex:(binIndex+1)*2**logIndex]=1
#             basisList[count, :]=tempBasis
#             count+=1;
#     return basisList

# This version allows logBinUpTo to be smaller than log2(lengthOfChip)
def generateBasis_V2(lengthOfChip, logBinUpTo):
    basisList=np.zeros((coefficientSize_V2(lengthOfChip, logBinUpTo),lengthOfChip));
    count=0;
    for logIndex in range(0,logBinUpTo+1):
        for binIndex in range(0,lengthOfChip//(2**logIndex)):
            tempBasis=np.zeros(lengthOfChip);
            tempBasis[binIndex*2**logIndex:(binIndex+1)*2**logIndex]=1
            basisList[count, :]=tempBasis
            count+=1;
    return basisList

# Return total number of coefficients
def coefficientSize_V2(lengthOfChip, logBinUpTo):
    Size=0;
    for i in range(logBinUpTo+1):
        Size+= lengthOfChip // 2**i 
    return Size

# # 2^0 pixel bins, 2^1 pixel bins, ...
# def coefficientSize(logBinUpTo):
#     Size=0;
#     for i in range(logBinUpTo+1):
#         Size+=2**i 
#     return Size

def generateGT(numOfSamples, lengthOfChip):
    GT = np.random.rand(lengthOfChip, numOfSamples);
    GT = (GT/np.sum(GT,0))*100
    return GT # Shape of (lengthOfChip, numOfSamples)

def generateEstimatorList(targetBox, basisList, lengthOfChip, logBinUpTo):
    estimatorList=[];

    # Define the estimators based on a vector of length basisList.shape[0], 1 --> add, -1 --> minus

    # Single pixels first
    estimatorList.append(np.concatenate((targetBox,np.zeros(basisList.shape[0]-lengthOfChip))))

    # Going through each measurement, compensate for the difference with single pixel measurement
    differenceMatrix=targetBox-basisList

    # Starting from index=lengthOfChip onwards, they are the estimators needed
    for i in range(lengthOfChip,basisList.shape[0]):
        tempVector=np.zeros(coefficientSize_V2(lengthOfChip, logBinUpTo));
        tempVector[i]=1
        tempVector+=np.concatenate((differenceMatrix[i,:],np.zeros(basisList.shape[0]-lengthOfChip)))
        estimatorList.append(tempVector)
    return estimatorList

def binning(GT, binningSize, exposureTime=1):
    numOfSamples=GT.shape[1]
    return scipy.signal.convolve(GT,np.ones((binningSize, 1)),mode='valid')[::binningSize, :]*exposureTime

def addNoise(binnedGT, readNoiseSD):
    # assert np.min(binnedGT)>0;
    return np.random.poisson(np.abs(binnedGT)) + np.random.normal(loc = 0, scale=readNoiseSD, size=binnedGT.shape)

def findMVUE(fileName, targetBox=[1,1,1,1], lengthOfChip=4, logBinUpTo=None):

    if logBinUpTo==None:
        logBinUpTo=int(np.log2(lengthOfChip))

    # Development in progress

    basisList=generateBasis_V2(lengthOfChip, logBinUpTo);

    estimatorList=generateEstimatorList(targetBox, basisList, lengthOfChip, logBinUpTo);

    # Declare a sympy array, storing all the weights
    weightArray=symarray('w',len(estimatorList))
    weightArray[-1]=1-(np.sum(weightArray)-weightArray[-1]) ### added 

    # Multiply each estimator by a weight, then sum up all the estimators
    allEstimator=np.zeros(coefficientSize_V2(lengthOfChip, logBinUpTo))
    for i in range(len(estimatorList)):
        allEstimator=allEstimator+estimatorList[i]*weightArray[i]
        
    # No covariance between different estimators, since they are independent measurements
    varianceCoefArray=allEstimator*allEstimator
    # Lagrangian is then the variance plus lambda multiplied by (\sum_i w_i -1)

    # Declare a sympy array, storing all the weights
    weightArray=symarray('w',len(estimatorList))
    weightArray[-1]=1-(np.sum(weightArray)-weightArray[-1]) ### added 

    # Multiply each estimator by a weight, then sum up all the estimators
    allEstimator=np.zeros(coefficientSize_V2(lengthOfChip, logBinUpTo))
    for i in range(len(estimatorList)):
        allEstimator=allEstimator+estimatorList[i]*weightArray[i]

    # No covariance between different estimators, since they are independent measurements
    varianceCoefArray=allEstimator*allEstimator
    # Lagrangian is then the variance plus lambda multiplied by (\sum_i w_i -1)

    # A matrix storing the langrangian differentiated w.r.t. each of the w_s
    diffMatrix=np.empty((len(allEstimator),len(weightArray)-1),dtype=object); 
    for i in range(len(allEstimator)):
        for j in range(len(weightArray)-1):
            diffMatrix[i,j]=diff(varianceCoefArray[i],weightArray[j])

    # return diffMatrix
            
    # print(diffMatrix)

    # Define the variance variables for each of the "measurement estimator"
    varianceArray=symarray('V',(len(allEstimator)))

    s = symbols("s")

    finalSolve=np.dot(diffMatrix.T,varianceArray)

    t=time.time();

    finalSolveWeightCoeffMatrix=np.empty((len(finalSolve),len(weightArray)-1),dtype=object);
    for i in range(len(finalSolve)):
        for j in range(len(weightArray)-1):
            finalSolveWeightCoeffMatrix[i,j]=diff(finalSolve[i],weightArray[j])
            
    constantOnTheRight=-finalSolve;
    subsDict={weight: 0 for weight in weightArray}
    for w1 in range(len(weightArray)-1):
        constantOnTheRight[w1]=constantOnTheRight[w1].subs(subsDict)

    finalSolveWeightCoeffMatrixSympy=Matrix(finalSolveWeightCoeffMatrix);
    print(time.time()-t)

    saveDict={'finalSolveWeightCoeffMatrixSympy':finalSolveWeightCoeffMatrixSympy, 'constantOnTheRight':Matrix(constantOnTheRight), 'varianceArray':varianceArray,'basisList':basisList,'weightArray':weightArray,'estimatorList':estimatorList}

    # Cannot pickle lambda function?
    # saveDict={'matrixFunc':matrixFunc, 'COTRFunc':COTRFunc, 'varianceArray':varianceArray,'basisList':basisList,'weightArray':weightArray,'estimatorList':estimatorList}

    with open(fileName, 'wb') as handle:
        pickle.dump(saveDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return saveDict
    
    # return {'estimatorList':estimatorList, 'weightArray':weightArray, 'varianceArray':varianceArray, 'soln':soln}

    # for key, value in soln.items():
    #     print(key, '= ', value)

    

def replacingVariance(varianceMapping, estimatorList, weightArray, varianceArray, soln):
    variancePlacerList=symarray('V_temp',(len(varianceArray)))
    placerReplacements=[(varianceArray[i],variancePlacerList[i]) for i in range(len(varianceArray))];
    backReplacements=[(variancePlacerList[i], varianceArray[varianceMapping[i]]) for i in range(len(varianceArray))];
    newSoln={};
    for i in range(len(weightArray)-1):
        newSoln[weightArray[i]]=soln[weightArray[i]].subs(placerReplacements);
        newSoln[weightArray[i]]=newSoln[weightArray[i]].subs(backReplacements);
#     varianceMapping=[1,0,2,3,4,5,6];
    target=np.asarray(estimatorList);
    newTarget=target.copy();
    for i in range(len(varianceMapping)):
        newTarget[:,i]=target[:,varianceMapping[i]];
    estimatorList=list(newTarget)
    return [estimatorList, weightArray, varianceArray, newSoln]

def loadBoxes(boxes, fileLocation):
    # t=time.time();
    dataDict={};
    for box in boxes:
        fileName=str(box).replace('1',' 1').replace('0',' 0');
        fullFileName=os.path.join(fileLocation, fileName);
        with open(fullFileName,'rb') as file:
            tempPickle = (pickle.load(file));
            varianceArray=tempPickle['varianceArray'];
            tempDict={'matrixFunc': (sp.lambdify(varianceArray, tempPickle['finalSolveWeightCoeffMatrixSympy'],[{'ImmutableMatrix':numpy.matrix},"numpy"])),
                     'COTR': (sp.lambdify(varianceArray, tempPickle['constantOnTheRight'],[{'ImmutableMatrix':numpy.matrix},"numpy"])),
                     'estimatorArray': np.asarray(tempPickle['estimatorList']),
                     'varianceArray':varianceArray};
        # Use tuple version of the ndarray as key
        dataDict[tuple(box.tolist())]=tempDict;
    # print(time.time()-t)
    return dataDict

def newLoadBoxes(boxes, fileLocation):
	# t=time.time();
    dataDict={};
    for box in boxes:
        fileName=str(box).replace('\n','_');
        fullFileName=os.path.join(fileLocation, fileName);
        with open(fullFileName,'rb') as file:
            tempPickle = (pickle.load(file));
            varianceArray=tempPickle['varianceArray'];
            tempDict={'matrixFunc': (sp.lambdify(varianceArray, tempPickle['finalSolveWeightCoeffMatrixSympy'],[{'ImmutableMatrix':numpy.matrix},"numpy"])),
                     'COTR': (sp.lambdify(varianceArray, tempPickle['constantOnTheRight'],[{'ImmutableMatrix':numpy.matrix},"numpy"])),
                     'estimatorArray': np.asarray(tempPickle['estimatorList']),
                     'varianceArray':varianceArray};
        # Use tuple version of the ndarray as key
        dataDict[tuple(box.tolist())]=tempDict;
    # print(time.time()-t)
    return dataDict

def getdyadicData_ET(GT, binningList, exposureTimeList,readNoiseSD):
    assert len(binningList)==len(exposureTimeList);
    dyadicData_ET=np.zeros((0,GT.shape[1]));
    for i in range(len(binningList)):
        dyadicData_ET = np.vstack((dyadicData_ET, 
                                 addNoise(binning(GT=GT, binningSize=binningList[i], 
                                         exposureTime=exposureTimeList[i]), readNoiseSD)/exposureTimeList[i]))
    return dyadicData_ET

def printMeanNSTD(index, *args):
    for i in range(0,len(args)):
        print('{:06.4f}'.format(np.mean(args[i][index,:])),
        '{:06.4f}'.format(np.std(args[i][index,:])))
        
def estimateVariance(dyadicData, lengthOfChip, logBinUpTo, binningList, exposureTimeList, readNoiseSD):
    varianceEstimate_ET = np.zeros(dyadicData.shape);
    assert len(binningList) == len(exposureTimeList);
    assert logBinUpTo + 1 == len(binningList);
    count=0;
    for logIndex in range(logBinUpTo+1):
        newCount=count+lengthOfChip//(2**logIndex);
        varianceEstimate_ET[count:newCount,:] = (dyadicData[count:newCount,:] / exposureTimeList[logIndex] + 
                               readNoiseSD**2 / exposureTimeList[logIndex]**2);
        count=newCount;
    return varianceEstimate_ET

def runImprovNum_Intuitive_ExposureTime(dyadicData, dataForVarianceEstimate, numOfSamples, basisDict, basisList, readNoiseSD, exposureTime):
    # t=time.time()
    varianceArray=basisDict[tuple(basisList[0].tolist())]['varianceArray'];
    len1=len(basisDict[tuple(basisList[0].tolist())]['COTR'](*np.zeros(len(varianceArray))));
    len2=len(basisDict[tuple(basisList[0].tolist())]['estimatorArray']);
    len3=dyadicData.shape[0];
    len4=len(basisList)
    newDyadicData=np.empty((len3,numOfSamples), float)
    for i in range(numOfSamples):
        dataMeasured=dyadicData[:,i];
        data=dataForVarianceEstimate[:,i];
        allWeights=np.empty((len1
                            ,len4), float);
        allEstimators=np.empty((len2
                               ,len4), float);
        count=0;
        for basis in basisList:
            tupleBasis=tuple(basis.tolist());
            coefMatrix=basisDict[tupleBasis]['matrixFunc'](*(data));
            COTR=basisDict[tupleBasis]['COTR'](*(data)) # Constant on the right
            weight = np.linalg.solve(coefMatrix,COTR);
            allWeights[:,count] = weight.flatten();
            estimatorValues=(basisDict[tupleBasis]['estimatorArray'] @ dataMeasured).reshape((-1,1))
            allEstimators[:,count]=estimatorValues.flatten();
            count+=1;
        allWeights=np.vstack((allWeights,1-np.sum(allWeights,0)));
        newData=np.sum(np.multiply(allEstimators, allWeights),0)
        newDyadicData[:,i]=newData
    # print(time.time()-t)
    return np.asarray(newDyadicData)

def runMVUE_n_times(n, dyadicData_ET, numOfSamples, basisDict, basisList, readNoiseSD, 
                    exposureTimeList, lengthOfChip, logBinUpTo, binningList):
    outputData=[dyadicData_ET];
    varianceEstimates=[estimateVariance(dyadicData_ET, lengthOfChip, logBinUpTo, 
                 binningList, exposureTimeList, readNoiseSD)];
    for i in range(n):
        # Note! Always use the same data dyadicData_ET as estimation! But varianceEstimates get improved everytime!
        # NEVER USE updated VarianceEstimates as new dyadicData_ET! --> This violates the underlying error distribution!!!
        outputData.append(runImprovNum_Intuitive_ExposureTime
                          (dyadicData_ET, varianceEstimates[i], numOfSamples, 
                          basisDict, basisList, readNoiseSD, exposureTimeList))
        varianceEstimates.append(estimateVariance(outputData[i+1], lengthOfChip, logBinUpTo, 
                 binningList, exposureTimeList, readNoiseSD));
    return outputData, varianceEstimates

# Get a list of mean absolute error percentage, then plot a graph
def storeShiftedBoxResults(shiftedDyadicData, SP_Data, GT, exposureTimeList):
    dyadicMAEList=[];
    SPMAEList=[];
    for lengthOfROI in range(10,16):
        targetBoxes=getShiftBoxes(16,lengthOfROI);
        for i in range(len(targetBoxes)):
            dyadicMAEList.append(np.mean(np.abs((shiftedDyadicData[lengthOfROI][i,:] 
                                                                         - np.dot(targetBoxes[i].T,GT))/np.dot(targetBoxes[i].T,GT))
                                                                  *100))
            SPMAEList.append(np.mean(np.abs((np.dot(targetBoxes[i].T,SP_Data/np.sum(np.asarray(exposureTimeList))) - 
                                            np.dot(targetBoxes[i].T,GT))/np.dot(targetBoxes[i].T,GT))
            *100))
    return dyadicMAEList, SPMAEList

# # For Mean Squared Error Percentage
# def storeShiftedBoxResults(shiftedDyadicData, SP_Data, GT, exposureTimeList):
#     dyadicMSEList=[];
#     SPMSEList=[];
#     for lengthOfROI in range(10,16):
#         targetBoxes=getShiftBoxes(16,lengthOfROI);
#         for i in range(len(targetBoxes)):
#             dyadicMAEList.append(np.mean(np.abs((dyadicData[lengthOfROI][i,:] 
#                                                                          - np.dot(targetBoxes[i].T,GT))/np.dot(targetBoxes[i].T,GT))
#                                                                   *100))
#             SPMAEList.append(np.mean(np.abs(np.dot(targetBoxes[i].T,SP_Data/np.sum(np.asarray(exposureTimeList))) - 
#                                             np.dot(targetBoxes[i].T,GT)/np.dot(targetBoxes[i].T,GT)*100)))
    return dyadicMAEList, SPMAEList

def printShiftedBoxResults(dyadicData, SP_Data, GT, exposureTimeList):
    for lengthOfROI in range(10,16):
        targetBoxes=getShiftBoxes(16,lengthOfROI);
        for i in range(len(targetBoxes)):
            print('for dyadic:', targetBoxes[i])
            print('Mean abs error perc = {:.2f} %'.format(np.mean(np.abs((dyadicData[lengthOfROI][i,:] 
                                                                         - np.dot(targetBoxes[i].T,GT))/np.dot(targetBoxes[i].T,GT))
                                                                  *100)))
            print('for SP:')
            print('Mean abs error perc = {:.2f} %'.format(np.mean(np.abs((np.dot(targetBoxes[i].T,
                                                                                (SP_Data/np.sum(np.asarray(exposureTimeList))))
                                                                         - np.dot(targetBoxes[i].T,GT))/np.dot(targetBoxes[i].T,GT)*100))))
            
def printMeanNSTD(index, *args):
    for i in range(0,len(args)):
        print('{:06.4f}'.format(np.mean(args[i][index,:])),
        '{:06.4f}'.format(np.std(args[i][index,:])))
        
def computeTargetLengths(dyadicData, dataForVarianceEstimate, numOfSamples, readNoiseSD, dataDict, i, lengthOfChip): # ComputeTargetLength
    return computeTargetLength_ExposureTime(dyadicData, dataForVarianceEstimate, 
                                                 numOfSamples, readNoiseSD, dataDict, i, lengthOfChip)    

# Pass in a dictionary which stores the pickles
# dataDict[num] stores the corresponding functions for lengthOfROI = num
def computeTargetLength_ExposureTime(dyadicData, dataForVarianceEstimate, numOfSamples, readNoiseSD, dataDict, lengthOfROI, lengthOfChip):
    # t=time.time()
    targetBoxes=getShiftBoxes(lengthOfChip,lengthOfROI);
    data=dataDict[lengthOfROI];
    targetBoxResult=np.empty((len(targetBoxes),0), float)
    for i in range(numOfSamples): # numOfSamples
        dataMeasured=dyadicData[:,i];
        dataVEstimate=dataForVarianceEstimate[:,i];
        # make it better
        allWeights=np.empty((data[tuple(targetBoxes[0].tolist())]['COTR'](*dataVEstimate).shape[0],0), float);
        allEstimators=np.empty((data[tuple(targetBoxes[0].tolist())]['estimatorArray'].shape[0],0), float);
        for targetBox in targetBoxes:
            targetBox=tuple(targetBox.tolist())
            coefMatrix=data[targetBox]['matrixFunc'](*(dataVEstimate)); # removed SD **2
            COTR=data[targetBox]['COTR'](*(dataVEstimate)) # Constant on the right
            weight = np.linalg.solve(coefMatrix,COTR)
            allWeights=np.hstack((allWeights,weight))
            estimatorValues=(data[targetBox]['estimatorArray'] @ dataMeasured).reshape((-1,1))
            allEstimators=np.hstack((allEstimators,estimatorValues))
        allWeights=np.vstack((allWeights,1-np.sum(allWeights,0)));
        newData=np.sum(np.multiply(allEstimators, allWeights),0)
        targetBoxResult=np.hstack((targetBoxResult, newData.T))
    # print(time.time()-t)
    return np.asarray(targetBoxResult)