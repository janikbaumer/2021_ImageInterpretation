import numpy as np
import math

#A function to compute root mean square error (RMSE)
def calculate_RMSE(Pred, GT):
    if np.shape(GT) == np.shape(Pred): #First check that the inputs are compatible
        n = np.shape(GT)[0] * np.shape(GT)[1]  #Total number of elements or pixels being compared
        RMSE = np.sqrt((1/n)*(np.sum((Pred - GT)**2))) #RMSE calculation
        return RMSE
    else:
        print('Error: Incompatible Matrices')
        
#A function to compute mean absolute error (MAE)
def calculate_MAE(Pred, GT):
    if np.shape(GT) == np.shape(Pred): #First check that the inputs are compatible
        n = np.shape(GT)[0] * np.shape(GT)[1]  #Total number of elements or pixels being compared
        MAE = (1/n)*(np.sum(np.abs(Pred - GT))) #MAE calculation
        return MAE
    else:
        print('Error: Incompatible Matrices')

if __name__ == '__main__':
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
