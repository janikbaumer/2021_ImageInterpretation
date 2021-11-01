# Research Notes
## Some notes I compiled from basic research conducted on the subject of classifiers

### Supervised Classifier Training
We want to train an "**Eager Learner**" algorithm, which basically means we let the algorithm train with a supervised set of training data, to tune the paraneters of the model which we later use to classify "unknown" data.
This in contrast to a "**Lazy Learner**", which compares test data to the training data when tested, in order to classify based on what in the training data the test data most ressembles.

### Algorithm Example
**Naive Bayes**
This is a probabilistic classifier, based on the hypothesis that the data attributes are independent. 
In practice, this method seems to work pretty well even when this assumption isn't true. 
Could be a good **Baseline method**?

## Method Evaluation
Potential metrics?

### Cross-Validation
Dividing the training data into k folds, then performing k iterations of the training, each time changing which fold is the validation fold, and in the end averaging out the parameters.
This method is useful to mitigate **Over-Fitting**.

### Confusion Matrix
A matric quantifiying truths and predicitons. Overview of false-positives, false-negatives etc. In our case it is a 3x3 matrix, because we have three classes.

### Precision-Recall
Calculate the two metrics from the confusion matrix, based on retrieved and relevant instances. (THIS SECTION NEEDS FLESHING OUT)

### ROC curve (Reciever Operator Call)
Useful for visually comparing different methods. Plots the methods true-positive rate versus false-positive rate. The area underneath the curve is the accuracy, which a maximum accuracy of 1 possible. The closer the curve is to the diagonal, the less accurate it is.


## Sources
Overview article: https://towardsdatascience.com/machine-learning-classifiers-a5cc4e1b0623
