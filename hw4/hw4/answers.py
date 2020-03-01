r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=64,
              gamma=0.99,
              beta=0.5,
              learn_rate=3.1*1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======

    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=64,
              gamma=0.99,
              beta=0.333,
              delta=1.5,
              learn_rate=1.7*1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======

    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

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

$v_\pi(s)$ is expressed in terms of $q_\pi(s,a)$ in such a way that $v_\pi(s) = q_\pi(s,a) - advantage$

The advantage function's purpose is to capture how better an action is compared to the other possible actions -
at a given state, and of course the value function's (v) purpose is to capture how good it is for our agent to
be in this state.

Now, the reason that when using the estimated q-values as regression targets for our state-values (v) we get
a valid approximation is that instead of having the critic (in the AAC) to learn the q-values, we make it learn
the values of the different advantages. This causes the evaluation of a certain action to be based not only on
how good the action is, but also how better it can get, which is not just a valid approximation but an improved one
over the previous method. By that we are stabilizing the model and reducing the high variance of those networks. 


"""


part1_q3 = r"""
**Your answer:**

We got 4 graphs in the first experiment run, and following are some insights derived from them:

- The losses of the models are want to go towards 0
- they are doing a good job since the trend in all of them is mostly increasing
- we can see that the different losses
- in loss_p, combined achieves the best results as proper to an improved model
- in the loss entropy graph, the entropy line does best
- 


- aac is generally doing better than the policy gradient. one of the reasons is what we stated in q2, which
makes the aac model an improved one over the pg 




"""
