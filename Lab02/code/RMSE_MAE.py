import numpy as np
import math

#A function to compute root mean square error (RMSE)
def calculate_RMSE(Pred, GT):
    if np.shape(GroundTruth) == np.shape(Predictions): #First check that the inputs are compatible
        N = np.shape(GroundTruth)[0] * np.shape(GroundTruth)[1]  #Total number of elements or pixels being compared
        RMSE = np.sqrt((1/N)*(np.sum((Predictions - GroundTruth)**2))) #RMSE calculation
        return RMSE
    else:
        print('Error: Incompatible Matrices')
        
#A function to compute mean absolute error (MAE)
def calculate_MAE(Pred, GT):
    if np.shape(GroundTruth) == np.shape(Predictions): #First check that the inputs are compatible
        N = np.shape(GroundTruth)[0] * np.shape(GroundTruth)[1]  #Total number of elements or pixels being compared
        MAE = (1/N)*(np.sum(np.abs(Predictions - GroundTruth))) #MAE calculation
        return MAE
    else:
        print('Error: Incompatible Matrices')

GroundTruth = np.array([[1, 9], [6, 8]])
print(GroundTruth)
print('')
Predictions = np.array([[5, 2], [4, 3]])
print(Predictions)
print('')

RMSE = calculate_RMSE(Predictions, GroundTruth)
print('RMSE')
print(RMSE)
print('')

MAE = calculate_MAE(Predictions, GroundTruth)
print('MAE')
print(MAE)
print('')
