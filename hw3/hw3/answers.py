r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 128
    hypers['seq_len'] = 4
    hypers['h_dim'] = 64
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 0.0001
    hypers['lr_sched_factor'] = 0.0001
    hypers['lr_sched_patience'] = 0.0001
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the entire text because texts in their
nature consist of words which are short character sequences, and therefore in real world (also done
by humans) in order to be able to predict the next character or few characters we want to train by looking
at a small text window rather than the entire corpus in whole. Note that a small text window
could also be of size 1, which means we train on one character in each iteration.

Additionally, we want to conclude relations between characters and between words and characters,
in order to predict the next most probable character, which will be harder to do if we look at the
entire corpus dataset as a whole.

"""

part1_q2 = r"""
**Your answer:**

The generated text clearly shows memory longer than the sequence length because the network was added
a memory component, which can remember inputs and then recall those inputs at later time stamps.
It looks also at states prior to the last one and not just the last one, hence preserving "memory"
which can be longer than the sequence length.

Explanation: much like LSTM's, the GRU's have a memory cell, and they solved the vanishing gradients
problem by introducing a new memory gate, that allowed hidden state to either pass through time (remembered)
or not (forgot). This enabled the modeling of sequence with much greater length than before.


"""

part1_q3 = r"""
**Your answer:**

Unlike previous tasks that we dealt with, in which the different batches did not have a specific order
to them, here the batches do have a specific order: we are analyzing text bits of a larger corpus, where
the learning order is important: we would like to learn the text in the order it appears in the corpus,
because eventually we would like to generate new characters and words. If we shuffle the order of batches,
we will 'confuse' the model to learn unrelated and unordered bits of text, which in turn will generate
new text in an unrelated and unordered manner, which is not what we desire.

"""

part1_q4 = r"""
**Your answer:**

We will answer all 3 sub-questions together, referring to temperature as 'temp.' or 'T':

The temp. hyperparameter of networks is used in order to control the randomness of predictions by scaling
the logits before applying softmax. It increases the sensititvity to low probability candidates.
For $T = 1$ which is normally the default value, no scaling is being done and we get the value as is.

Using T that is lower than 1, for instance $T = 0.5$, the model computes the softmax on the scaled logits,
i.e in this case $logits/0.5$, which increases the logits values.
Next, performing softmax on larger values makes the LSTM more confident but more conservative in its samples.
'More confident' means less input is needed to activate the output layer, and 'more conservative' means
it is less likely to sample form unlikely candidates.

On the other hand, using T that is higher than 1, for instance $T = 1.5$, we get a softer probability
distribution over the classes, which in turn makes the RNN more diverse and more error-prone, meaning it
is more likely to pick lower probability candidates and makes more mistakes.

Lastly, the softmax function normalizes the candidates at each iteration by ensuring the network outputs
are all between 0 and 1.

We would like to be more confident in order to be less mistaken (even if it means we are more conservative),
because what matters most to us is the accuracy of the results, and therefore as explained earlier,
we are going to use temp. values that are lower than 1 rather than higher than 1

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ in the VAE model represents the variance in the normal distribution.
Generally, if $\sigma^2$ will have too big of a value, we will have to take a very large number of samples to find
out the true expectation. When the variance is so high, using it to learn different models becomes hard.

In good practices using VAE, we prefer a low-variance gradient estimator based on the reparametrization trick.
And so in general, in different examples we've seen, high variance tends to be worse and low variance tends to
be better, so we need to lower the value, but making sure it is not becoming too low.

"""

part2_q2 = r"""
**Your answer:**

1)

In the first place, we want encodings that are as close as possible to each other while maintaining being distinct
from each other. This allows smooth interpolation and the generation of samples.
The KL divergence loss's purpose is to ensure this, while the divergence value between two probability
distributions measures how much they are different from each other.
By minimizing this loss we are optimizing the prob. dist. parameters (among them - $\sigma$) to be as close as it
can to the target dist.

Regarding the reconstruction loss, it is usually either the MSE or CE loss between the output and the input.
Its purpose is to penalize the network for creating outputs that are different from the input. 

2)

The KL divergence encourages the encoder to distribute all input encodings in an even manner around the center
of the latent space. If it clusters them apart into specific regions and away from the center, it will get a penalty.

3)

The benefit of all the input encodings being evenly distributed around the center of the latent space is that unlike
when the encodings are densely placed randomly near the center of the latent space, where the decoder finds it
very hard to decode anything meaningful from the space, when the two losses are optimized together we maintain the
similarity of nearby encodings locally, and globally it is densely packed near the center of the latent space.
And then when we reach the equilibrium of the two losses, then when randomly generating a sample (sampling a vector from
the same prior dist.), the decoder can successfully decode it, with no gaps between clusters and a smooth mix
of features. 

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

During training, when sampling from the GAN, while optimizing the discriminator we are using real and fake input.
Both the real input and the fake input will create a loss, however we would like to maintain gradients in the
discriminator according to the real data and according to it **only**. The reason is that we want to improve our
discriminator by 'telling it' which is real data so it will know according to which data to perform the learning process.
The fake data is used only for testing the performance so far, and we do not want to update the discriminator according to it.
 
And so we maintain gradients according to the real data only. And this is the reason why we detach the fake input from
the generator while we update the discriminator, and this way the generator won't see the other gradients. This is the case
when we are discarding gradients. And so because of the detachment process no gradients will be created in the generator.

"""


part3_q2 = r"""
**Your answer:**

1) We would prefer not to stop training solely based on the fact that the generator loss is below some threshold.
The reason for that is that our eventual goal is for the generator to be good enough so that the discriminator
would not be able to differentiate between a real and a fake image. And so if we decide to halt according to the
generator's performance solely, we might still get a discriminator that in fact **can** perform the differentiation
and so we stopped before we had to.

In general it takes some time for GANs to train, and it might take some time for the losses and the generated images to
show improvement and progress, and the early stopping according to the generator's loss alone is thus less recommended.

2) If the generator's loss decreases it generally means that the generator is improving meaning that it is able to
produce better more real-looking images. If the discriminator's loss remains at a constant value, it generally means
that the discriminator's performance is neither improving nor getting worse, and that its ability to differentiate
between real and fake images remained the same. It might mean that the hyperparameters of the discriminator lead it
to 'get stuck' in a local minimum of a convex function, thus is is neither improving nor its performance is getting
worse. And the way to solve it other than to recheck the implementation is to tune again the hyperparameters.
It might also mean that there is a randomness component to its discrimination process.

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


