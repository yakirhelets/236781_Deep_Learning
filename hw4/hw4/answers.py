r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======

    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=1.,
              delta=1.,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hp


part1_q1 = r"""

Subtracting a baseline in the policy-gradient basically leaves us with the gained
value of the current action in comparison to the action that we would take in the average case.
Since the same action can receive different rewards (based on the state), the policy needs to
collect a lot of experience in order to converge, which contributes to a high variance.
Thus, using only the gained value (i.e subtracting the baseline) helps us to eliminate the
variance gained by being in different states.

Such method helps especially when there are multiple trajectories with positive rewards.
By subtracting the baseline, the rewards will be more zero-centered, thus less positive rewards.
That means that the policy-gradient will increase the probability of less trajectories. 

"""


part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
