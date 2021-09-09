from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]


def build_lstm(
        input_dim:int,
        hidden_dim:int,
        output_size:int,
        n_layers:int,
        seq_len:int):

    return nn.Sequential(
        nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True),
        extractlastcell(),
        nn.Linear(hidden_dim, output_size))

    # lstm_layer = nn.LSTM(input_dim,hidden_dim,n_layers,batch_first = True)
    # input =torch.randn(batch_size,seq_len,input_dim)
    # hidden_state = torch.randn(n_layers,batch_size,hidden_dim)
    # cell_state = torch.randn(n_layers,batch_size,hidden_dim)
    # hidden = (hidden_state,cell_state)
    # out,hidden = lstm_layer(input,hidden)




def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'relu',#'tanh',
        output_activation: Activation = 'identity',
        use_dropout=False,
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
        if use_dropout:
            layers.append(nn.Dropout(p=0.2))
    layers.append(nn.Linear(in_size, output_size))

    layers.append(output_activation)
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
