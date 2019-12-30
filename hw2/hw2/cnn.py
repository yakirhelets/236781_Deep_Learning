import torch
import itertools as it
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every

        additional_layers = N % P
        first_layers = N - 1 - additional_layers # Minus additional one bc adding it manually as the first

        layers.append(nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if P == 1:
            layers.append(nn.MaxPool2d(kernel_size=2))

        dim_count = 0

        for j in range(first_layers):
            layers.append(nn.Conv2d(self.channels[dim_count], self.channels[dim_count+1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            dim_count += 1
            if (dim_count+1) % int(N/P) == 0:
                layers.append(nn.MaxPool2d(kernel_size=2))

        for i in range(additional_layers):
            layers.append(nn.Conv2d(self.channels[dim_count], self.channels[dim_count+1], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            dim_count += 1

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        M = len(self.hidden_dims)
        N = len(self.channels)
        P = self.pool_every

        ratio = N / P
        for i in range(int(ratio)):
            in_h /= 2
            in_w /= 2

        in_feats = int(in_h) * int(in_w) * self.channels[-1]

        layers.append(nn.Linear(in_feats, self.hidden_dims[0]))
        layers.append(nn.ReLU())

        for i in range(1, M):
            layers.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[M-1], self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        y = self.feature_extractor(x)
        y = y.view(y.shape[0], -1)
        out = self.classifier(y)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        L = len(channels)

        main_layers_list = []
        main_layers_list.append(nn.Conv2d(in_channels, channels[0], kernel_size=kernel_sizes[0], padding=(int((kernel_sizes[0]-1)/2))))
        main_layers_list.append(nn.Dropout2d(dropout))
        main_layers_list.append(nn.BatchNorm2d(channels[0])) if batchnorm else None
        main_layers_list.append(nn.ReLU())

        for c in range(L - 1):
            main_layers_list.append(nn.Conv2d(channels[c], channels[c+1], kernel_size=kernel_sizes[c+1], padding=(int((kernel_sizes[c+1]-1)/2))))
            if c != L-2:
                main_layers_list.append(nn.Dropout2d(dropout))
                main_layers_list.append(nn.BatchNorm2d(channels[c+1])) if batchnorm else None
                main_layers_list.append(nn.ReLU())

        sc_layers_list = []
        if in_channels != channels[L-1]:
            sc_layers_list.append(nn.Conv2d(in_channels, channels[L-1], kernel_size=1, bias=False))
        else:
            sc_layers_list.append(nn.Identity())

        self.main_path = nn.Sequential(*main_layers_list)
        self.shortcut_path = nn.Sequential(*sc_layers_list)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every

        additional_layers = N % P

        layers.append(ResidualBlock(in_channels, self.channels[0:P], [3] * P))
        layers.append(nn.MaxPool2d(kernel_size=2))

        dim_count = 0

        for i in range(int((N-P) / P)):
            # res net
            layers.append(ResidualBlock(self.channels[dim_count], self.channels[(i+1)*P:(i+2)*P], [3]*P))
            layers.append(nn.MaxPool2d(kernel_size=2))
            dim_count += P

        last_ch_used = N - additional_layers - 1

        if additional_layers != 0:
            # Final res net from first that was not covered to last
            layers.append(ResidualBlock(self.channels[last_ch_used], self.channels[last_ch_used+1:], [3]*(len(self.channels[last_ch_used+1:]))))


        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        layers.append(nn.Conv2d(in_channels, 32, kernel_size=3, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=2))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=2))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Conv2d(128, 64, kernel_size=3, padding=2))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Conv2d(64, 32, kernel_size=3, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []

        layers.append(nn.Linear(3200, 100))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(100, 200))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(200, 400))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(400, 200))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(200, 100))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(100, self.out_classes))

        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================
