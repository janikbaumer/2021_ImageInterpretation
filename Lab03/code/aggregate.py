import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix

#Inside loop, for each iteration, call scikit learn confusion matrix function using predictions and ground truth
#cm = confusion_matrix(y_true, y_pred, labels=["class 1", "class 2", "class ...", "class n"])
#Add up confusion matrices from each iteration (element wise addition)
#cm = cm + this_cm
#Should have dimensions (num_classes x num_classes)

#Example of a possible result:
"""
cm = np.array([[150,12,5,25], [21,730,0,30], [2,1,83,4], [17,5,0,350]])
print(cm)

#Plot the confusion matrix:
x_axis_labels = ['a', 'b', 'c', 'd'] # labels for x-axis
y_axis_labels = ['a', 'b', 'c', 'd'] # labels for y-axis
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show()
"""

#Statistics and metrics
def calc_metrics(cm):
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*TP/(2*TP + FP +FN)
    overall_accuracy = (sum(TP) + sum(TN))/(sum(TP) + sum(FP) + sum(FN) + sum(TN))
    return precision, recall, f1, overall_accuracy
"""
print("TP per class:", TP)
print("FP per class:", FP)
print("FN per class:",FN)
print("TN per class:",TN)
print("Precision score per class", precision)
print("Recall score per class", recall)
print("F1 score per class", f1)
print("Overall accuracy for all classes combined:", overall_accuracy)
"""