import re

import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    char_to_idx = {}
    idx_to_char = {}

    chars = []
    # Get all unique chars
    [chars.append(c) for c in text if c not in chars]
    chars.sort()

    dict_counter = 0
    # Build the dictionaries
    for c in chars:
        char_to_idx[c] = dict_counter
        idx_to_char[dict_counter] = c
        dict_counter += 1

    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = ""
    n_removed = 0

    # Go over the text and add to text_clean what's not in chars_to_remove
    for c in text:
        if c not in chars_to_remove:
            text_clean += c
        else:
            n_removed += 1

    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    N = len(text)
    D = len(char_to_idx)

    result = torch.zeros([N, D], dtype=torch.int8)

    char_count = 0
    for c in text:
        result[char_count, char_to_idx[c]] = 1
        char_count += 1

    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    N = embedded_text.shape[0]
    D = embedded_text.shape[1]
    result = ""

    for i in range(N):
        for j in range(D):
            if embedded_text[i, j] == 1:
                result += (idx_to_char[j])
                break

    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    N = int(len(text) / seq_len)
    S = seq_len

    # 1
    embedded_text = chars_to_onehot(text, char_to_idx)
    V = embedded_text.shape[1]
    samples = torch.zeros([N, S, V], dtype=torch.int8, device=device)
    labels = torch.zeros([N, S], dtype=torch.int8, device=device)

    # TODO add ignoring the last char
    #2
    for i in range(N):
        samples[i,:,:] = embedded_text[i*S:(i*S)+S, :]

    #3
    for i in range(N):
        # get substring from i to i+1 to S+1 of text
        sub = text[(i*S)+1:((i*S)+1+S)]
        sub_indices = []
        for j in range(len(sub)):
            sub_indices.append(char_to_idx[sub[j]])
        sub_tensor = torch.as_tensor(sub_indices, dtype=torch.int8, device=device)
        # append to labels as row tensor
        labels[i, :] = sub_tensor

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======

    e_y_t = torch.exp(y / temperature)
    sigma_e_y_t = 0
    for i in y:
        sigma_e_y_t += torch.exp(i / temperature)
    result = e_y_t / sigma_e_y_t

    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======

    window_size = len(start_sequence)
    h = None
    seq = out_text[-window_size:]

    for c in range(n_chars):
        x = chars_to_onehot(seq, char_to_idx).unsqueeze(dim=0).float()
        y, h = model(x, h)
        soft = hot_softmax(torch.squeeze(y)[-1], temperature=T) # Taking the last output char only
        new_char_index = torch.multinomial(soft, 1) # Sampling one new char

        soft_right_shape = soft.unsqueeze(dim=0)
        tmp_tensor = torch.zeros_like(soft_right_shape, requires_grad=False)
        tmp_tensor[0, new_char_index] = 1
        new_char = onehot_to_chars(tmp_tensor, idx_to_char)
        out_text += new_char
        seq = out_text[-window_size:]

    out_text = out_text[6:]

    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents  one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of indices is takes, samples in the same index of
        #  adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        print(len(self.dataset))
        print(self.batch_size)
        idx = []
        for i in range(int(len(self.dataset)/self.batch_size)):
            for j in range(self.batch_size):
                idx.append(j + (self.batch_size * i))
        # TODO make sure correct

        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of output dimensions (at each timestep).
        :param n_layers: Number of layers in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======

        self.dropout = dropout

        # Input layer
        self.add_module("Layer_0_update_x", nn.Linear(in_dim, h_dim))
        self.add_module("Layer_0_reset_x", nn.Linear(in_dim, h_dim))
        self.add_module("Layer_0_hidden_x", nn.Linear(in_dim, h_dim))

        self.add_module("Layer_0_update_h", nn.Linear(h_dim, h_dim, bias=False))
        self.add_module("Layer_0_reset_h", nn.Linear(h_dim, h_dim, bias=False))
        self.add_module("Layer_0_hidden_h", nn.Linear(h_dim, h_dim, bias=False))

        # self.add_module("Dropout_0", nn.Dropout(dropout))

        Wxz = nn.Linear(h_dim, h_dim)
        self.layer_params.append(Wxz)
        Wxr = nn.Linear(h_dim, h_dim)
        self.layer_params.append(Wxr)
        Wxg = nn.Linear(h_dim, h_dim)
        self.layer_params.append(Wxg)

        Whz = nn.Linear(h_dim, h_dim, bias=False)
        self.layer_params.append(Whz)
        Whr = nn.Linear(h_dim, h_dim, bias=False)
        self.layer_params.append(Whr)
        Whg = nn.Linear(h_dim, h_dim, bias=False)
        self.layer_params.append(Whg)



        # Middle layers
        for i in range(1, self.n_layers):

            # Adding params according to the formulas in the notebook, and adding modules accordingly

            self.add_module(f"Layer_{i}_update_x", Wxz)
            self.add_module(f"Layer_{i}_reset_x", Wxr)
            self.add_module(f"Layer_{i}_hidden_x", Wxg)

            self.add_module(f"Layer_{i}_update_h", Whz)
            self.add_module(f"Layer_{i}_reset_h", Whr)
            self.add_module(f"Layer_{i}_hidden_h", Whg)

            self.add_module(f"Dropout_{i}", nn.Dropout(dropout))

        W_last = nn.Linear(h_dim, out_dim)
        self.layer_params.append(W_last)
        self.add_module(f"Last Layer", W_last)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======

        layer_output_params = []

        for i in range(seq_len):

            x = layer_input[:, i, :]
            h_t_minus_1 = layer_states[0]

            z = nn.Sigmoid()(nn.Linear(self.in_dim, self.h_dim)(x) + self.layer_params[3](h_t_minus_1))
            r = nn.Sigmoid()(nn.Linear(self.in_dim, self.h_dim)(x) + self.layer_params[4](h_t_minus_1))
            g = nn.Tanh()(nn.Linear(self.in_dim, self.h_dim)(x) + self.layer_params[5](r * h_t_minus_1))
            h_t = z * h_t_minus_1 + (1 - z) * g

            layer_states[0] = nn.Dropout(self.dropout)(h_t)  # adding dropout layer

            # x = layer_input[:,i+1,:]
            x = h_t

            for j in range(1, self.n_layers):

                h_t_minus_1 = layer_states[j]

                z = nn.Sigmoid()(self.layer_params[0](x) + self.layer_params[3](h_t_minus_1))
                r = nn.Sigmoid()(self.layer_params[1](x) + self.layer_params[4](h_t_minus_1))
                g = nn.Tanh()(self.layer_params[2](x) + self.layer_params[5](r*h_t_minus_1))
                h_t = z * h_t_minus_1 + (1 - z) * g
                x = h_t

                layer_states[j] = nn.Dropout(self.dropout)(h_t) # adding dropout layer

            layer_output_params.append(self.layer_params[6](h_t))

        # Concatinating along the second dimension, dim=1
        layer_output = torch.stack(layer_output_params, dim=1)
        hidden_state = torch.stack(layer_states, dim=1)

        # ========================
        return layer_output, hidden_state
            