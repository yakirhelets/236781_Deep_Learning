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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
