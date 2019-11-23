r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
Increasing ```k``` leads to improved generalization for unseen data up to a certain point. Large values of ```k```
(up to the largest number which is the size of the dataset) will determine the class of the unseen
data according to most of the examples in the dataset (in extreme values), which will be incorrect
classification if the correct class is smaller in size. However, a value of ```k``` that is too small might
also be incorrect if the closest example belongs to the incorrect class (which can happen if the number
of examples is too small). Therefore the ideal value of ```k``` should be neither too small nor too large.
"""

part2_q2 = r"""
**Your answer:**
1. Training on the entire train-set with various models and then selecting the best model with respect
to train-set accuracy is bad practice, since it leads to overfitting on the training data. We will then select
a model that performs best on the training data, while it may very well be very wrong on new unseen data.

2. Training on the entire train-set with various models and then selecting the best model with respect
to test-set accuracy is somewhat better than (1), since we are determining the selected model according to
untouched test-set. However, the selected model is very much influenced by the selected test set.
Dividing the data differently might lead to completely different and inconsistent results.

Using K-fold CV solves both problems. Overfitting is reduced a lot with K-fold CV since the selected model
is determined by an average of the accuracy on different train-sets (each time a new train-set is selected),
and the model is not biased towards one selected train-set.
In addition, the selected model is not influenced by one test-set, since each time the model is being tested
on a different test-set, which leads to consistent results and lack of bias.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The selection of Delta > 0 is arbitrary for the SVM loss because of the following reason. The magnitude of
the weights matrix W has a direct effect on the scores in that the more we make the values inside W smaller,
the more the score differences will become smaller, and vice versa. Since we can reduce or enlarge the score
differences arbitrarily, and we do not care what exact value the margin between the scores will have,
we can set the value of Delta to also be arbitrary.
"""

part3_q2 = r"""
**Your answer:**
1. Looking at the images, we can recognize shapes of digits in the actual images that were formerly weights matrices.
The leftmost one represents the digit 0 in its shape, the second one resembles 1, the next resembles 2 etc.
We can see that the linear model changes the weights matrices so that certain values (colors) appear
close to where the line should be, and other values appear everywhere else. This way, the model is learning to
predict digits according to the weights matrices - that appear closer to the input.
If we look at the classification errors, we can see that the digit sample (that was predicted incorrectly)
resembles in its writing to the weights matrix that correspond to the incorrect classification, and thus the
model mistakenly predicted incorrectly. This phenomenon might happen when a sample has noise in it, and in this
case this noise simply means a digit that its writing might be somewhat unclear. For instance, we can see that
one digit was 9, and was predicted to be 8. Looking closely at it we reveal that the lower part of the 9 looks
almost to 8.

2. It is similar to KNN because with KNN we can see that examples that are closer to each other in their features
tend to be represented with certain values (colors). Same as here, where a new unseen digit (sample) that is
close to a certain weight image in the colors - is similar to it in the features, and therefore will be given
the same classification.
"""

part3_q3 = r"""
**Your answer:**
1. Based on the graph of the training set loss, we would say that the learning rate that we chose is good.
Receiving a training set loss that slowly converges to the minimum (it goes slowly but eventually it converges)
means that the learning rate that was chosen is a GOOD learning rate. In our case, the prediction did not reach
optimality, however the accuracy is very good (over 85%) and we can conclude that the learning rate was good
for the given number of epochs.
If we look at a learning rate that is TOO LOW, with the same number of epochs, we will get too small of a step
towards convergence, while it very well may be the case that because the step is too small - the loss will never
converge and will remain with a high value (just because there are not enough epochs. If there were enough,
it would have converged). If we look at a learning rate that is TOO HIGH, we will get too big of a step towards
convergence. In this case it very well may be the case that because the step is too big - the loss will never
converge and will remain with a high value (in this case, the epochs number might not matter, since the minimum
point will always be missed by the big step).

2. In general, the higher the training set accuracy, the higher the overfitting is, and vice versa. In addition,
the higher the test set accuracy, the better the generalization and the lower the overfitting is.
That is because if we make our model too predictive towards the training set, we lose the ability to generalize
and therefore the training set accuracy will be high but the test set accuracy will be low.
In our case, based on the graph of the training and test set accuracy, we would say that the model is first of
all neither highly overfitted to the training set nor highly underfitted to it, because in the former case
we will indicate a high training set accuracy result but low test set accuracy result, and in the latter case
the opposite, and we are not indicating either (neither of the graphs is low).
Regarding the other options, because we are noticing that the training set's accuracy goes a little bit above
the test set accuracy (in the graph), we can conclude that our model is slightly overfitted, since it performs
better on the training set than on the test set, which is a result of a small overfitting of the training process.
It means that our results on the training set are very good (in this case), but we are not able to generalize
**as well** and get high results like these - in the test set (although the results on the test set are still very
good as well).
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

In a residual plot, the ideal pattern would be that the entire data set points lie
on the line that represents y - y_hat = 0. This way the prediction for the entire
data set is correct and the error rate will be zero (However, oftentimes it is
a situation that is difficult to achieve).

If we look at the residual plot in the last part (Generalization), we can
see that the fitness level of the trained model is pretty high, and it describes
the real world pretty well given the fact that most of the test data points lie
close to the line y - y_hat = 0 (which indicates correct prediction). It means
we were able to generalize our model pretty well based on the training data.

Additionally, looking at the plot for the top-5 features and looking at the
final plot after CV, we can identify that the former is less accurate, since
more data points lie further from the correct prediction line, while the latter
indicates choosing the best hyperparameters and thus improving the model, and
therefore shows an improvement when more data points lie closer to the correct
prediction line.

"""

part4_q2 = r"""
**Your answer:**

1. ```np.linspace``` represents numbers that are spaced evenly on a linear scale,
while ```np.logspace``` represents them spaced evenly on a log scale.
Since the lambda parameter is being used in the code in a multiplication,
it makes more sense to define its range using logspace, because we would like
to examine subtle changes in lambda's values. Large changes in its value will
greatly affect the end result of the computation and it will be difficult therefore
to select a value that is ideal. We would like to fine-tune its value in order
to get the best result.
Additionally, lambda's range, unlike other hyperparameters ranges, is continuous,
which makes it more suitable for logspace scale because the ideal value most
certainly lies somewhere in between discrete values. Other parameters, such
as the 'degree' parameter in the code, must be taken from a linspace scale (or,
as done above, be defined by hand with a range of integers). 

2. By performing K-fold CV, we are fitting the data K times, since we are dividing
the data into K different folds while in each of the K iterations we are fitting
(K-1) parts of the data and predicting on the remaining part.
If we look at the code above, and including hyperparameters examination,
we have 3 options for the degree and 20 options for lambda, which make 60 different
options for parameters combinations. In addition, we are using 3-fold CV, and so
overall we are fitting 3 times on each of the parameters combination, to make
180 fittings in total (not including the final fit on the entire training set). 

"""

# ==============
