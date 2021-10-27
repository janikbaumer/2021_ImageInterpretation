import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

def calc_precision(TP, FP):
    precision = TP/(TP + FP)
    return precision

def calc_recall(TP, FN):
    recall = TP/(TP + FN)
    return recall
    
def calc_f1(TP, FP, FN):
    f1 = 2*TP/(2*TP + FP + FN)
    return f1
    
#A function to calculate and derive metrics and statistics from a confusion matrix
#Assuming  that input matrices are always of size 3 x 3
def compute_metrics(cmat): 
    
    cmatsum = np.sum(cmat)
    
    TP0 = cmat[0, 0]
    FP0 = cmat[1, 0] + cmat[2, 0]
    FN0 = cmat[0, 1] + cmat[0, 2]
    TN0 = cmatsum - (TP0 + FP0 + FN0)
    precision0 = calc_precision(TP0, FP0)
    recall0 = calc_recall(TP0, FN0)
    f10 = calc_f1(TP0, FP0, FN0)
    
    TP1 = cmat[1, 1]
    FP1 = cmat[0, 1] + cmat[2, 1]
    FN1 = cmat[1, 0] + cmat[1, 2]
    TN1 = cmatsum - (TP1 + FP1 + FN1)
    precision1 = calc_precision(TP1, FP1)
    recall1 = calc_recall(TP1, FN1)
    f11 = calc_f1(TP1, FP1, FN1)
    
    TP2 = cmat[2, 2]
    FP2 = cmat[0, 2] + cmat[1, 2]
    FN2 = cmat[2, 0] + cmat[2, 1]
    TN2 = cmatsum - (TP2 + FP2 + FN2)
    precision2 = calc_precision(TP2, FP2)
    recall2 = calc_recall(TP2, FN2)
    f12 = calc_f1(TP2, FP2, FN2)
    
    precisions = np.array([precision0, precision1, precision2])
    recalls = np.array([recall0, recall1, recall2])
    f1s = np.array([f10, f11, f12])
    
    sumTP = TP0 + TP1 + TP2
    sumFP = FP0 + FP1 + FP2
    sumFN = FN0 + FN1 + FN2
    sumTN = TN0 + TN1 + TN2
    overall_accuracy = (sumTP + sumTN)/(sumTP + sumFP + sumFN + sumTN)
    
    return overall_accuracy, precisions, recalls, f1s



### main

'''
for model in list(['gnb', 'sgdc']):
    
    print('MODEL: ', model, '\n')
    for i in range(5):
        FILEPATH = 'cm_full_' + str(model) + str(i+1) + '.csv'

        # evaluation gnb
        CM = np.genfromtxt(FILEPATH, delimiter=',')

        overall_accuracy, precisions, recalls, f1s = compute_metrics(CM)

        print(f'loop {i+1}')
        #print()
        #print('overall accuracy: \n', overall_accuracy)
        #print()
        #print('precisions: \n', precisions)
        #print()
        #print('recalls: \n', recalls)
        #print()
        #print('f1 scores: \n', f1s)

        print('avg of f1 scores: ', np.average(f1s))
        print('\n\n')
'''

TESTFILE = 'cm_full_gnb3_testSet.csv'
# evaluation gnb
CM = np.genfromtxt(TESTFILE, delimiter=',')

overall_accuracy, precisions, recalls, f1s = compute_metrics(CM)

print('test set results: \n')
print()
print('overall accuracy: \n', overall_accuracy)
print()
print('precisions: \n', precisions)
print()
print('recalls: \n', recalls)
print()
print('f1 scores: \n', f1s)

print('avg of f1 scores: ', np.average(f1s))
print('\n\n')