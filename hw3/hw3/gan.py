from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_channels = in_size[0]
        self.out_channels = 1024

        modules = []

        modules.append(
            nn.Conv2d(self.in_channels, 64, 3, 1))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(64, 64, 3, 1))
        modules.append(nn.ReLU())

        modules.append(nn.Conv2d(64, self.out_channels, 3, 1))

        self.model = nn.Sequential(*modules)
        self.model.to(device)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        batch_size = x.shape[0]
        # features = torch.tanh(self.model(x))
        features = self.model(x)
        features = features.view(batch_size, -1)
        print(features.shape)

        num_features = features.shape[1]
        # fc_hidden = [1]
        # print(x.shape)
        # features = torch.tanh(self.model_gan(x))

        # Creating a fc NN for the classification part

        modules = []
        # M = len(fc_hidden)
        # mlp = num_features
        # for idx in range(M):
        #     modules.append(nn.Linear(num_features, fc_hidden[idx]))
        #     mlp = fc_hidden[idx]
        #     # layers.append(nn.Dropout(p=0.1))
        #
        # modules.append(nn.Linear(mlp, 1))
        modules = [nn.Linear(num_features, 1)]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_fc = nn.Sequential(*modules).to(device)
        print(model_fc)

        # print(device)
        y = model_fc(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.featuremap_size = featuremap_size
        self.out_channels = out_channels

        modules = []
        modules.append(nn.ConvTranspose2d(self.z_dim, 24, self.featuremap_size, padding=0, stride=2))
        modules.append(nn.ReLU())

        modules.append(nn.ConvTranspose2d(24, 64, self.featuremap_size, padding=1, stride=2))
        modules.append(nn.ReLU())

        modules.append(nn.ConvTranspose2d(64, 128, self.featuremap_size, padding=1, stride=2))
        modules.append(nn.ReLU())

        modules.append(nn.ConvTranspose2d(128, 256, self.featuremap_size, padding=1, stride=2))
        modules.append(nn.ReLU())

        modules.append(nn.ConvTranspose2d(256, self.out_channels, 4, padding=1, stride=2))

        self.model = nn.Sequential(*modules)

        self.model.to(device)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======

        N = n
        Other_dims = self.z_dim

        # create a random tensor
        rand_tensor = torch.rand((N, Other_dims), device=device, requires_grad=with_grad)
        # create the samples with forward of self
        samples = self.forward(rand_tensor).detach() if with_grad is False else self.forward(rand_tensor)

        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        z.to(device)
        z = z.unsqueeze(-1)
        z = z.unsqueeze(-1)

        x = self.model(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    N = y_data.shape[0]
    noise_half = label_noise / 2
    lower_th = -noise_half
    upper_th = noise_half

    gen_data_labels = torch.FloatTensor(N).uniform_(lower_th, upper_th)
    real_data_labels = torch.FloatTensor(N).uniform_(lower_th, upper_th)

    # We are guaranteed that data_label == 0 or 1 due to the assert
    if data_label == 0:
        gen_data_labels += 1
    elif data_label == 1:
        real_data_labels += 1

    criterion = torch.nn.BCEWithLogitsLoss()

    loss_data = criterion(y_data, real_data_labels)
    loss_generated = criterion(y_generated, gen_data_labels)

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    N = y_generated.shape[0]

    gen_data_labels = torch.FloatTensor(N).uniform_(data_label) # All same value

    criterion = nn.BCEWithLogitsLoss()

    loss = criterion(y_generated, gen_data_labels)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dsc_model.to(device)

    y_pred = dsc_model(x_data)

    num_to_sample = y_pred.shape[0]
    samples = gen_model.sample(num_to_sample, False)
    generated = dsc_model(samples)

    dsc_loss = dsc_loss_fn(y_pred, generated)

    dsc_optimizer.zero_grad()
    dsc_loss.backward(retain_graph=True)
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen_model.to(device)

    y_pred  = gen_model(x_data)

    num_to_sample = y_pred.shape[0]
    samples = gen_model.sample(num_to_sample, True)
    generated = dsc_model(samples)

    gen_loss = gen_loss_fn(generated)

    gen_optimizer.zero_grad()
    gen_loss.backward(retain_graph=True)
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======

    saved_state = dict(gen_model=gen_model.state_dict(),
                       dsc_losses=dsc_losses,
                       gen_losses=gen_losses)
    torch.save(saved_state, checkpoint_file)
    print(f'*** Saved checkpoint {checkpoint_file} ')

    saved = True
    # ========================

    return saved
