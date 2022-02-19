import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from typing import Tuple, List


def pack_padded_sequence(inputs, lengths):
    """Returns packed speechbrain-formatted tensors.

    Arguments
    ---------
    inputs : torch.Tensor
        The sequences to pack.
    lengths : torch.Tensor
        The length of each sequence.
    """
    lengths = (lengths * inputs.size(1)).cpu()
    return torch.nn.utils.rnn.pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False
    )


def pad_packed_sequence(inputs):
    """Returns speechbrain-formatted tensor from packed sequences.

    Arguments
    ---------
    inputs : torch.nn.utils.rnn.PackedSequence
        An input set of sequences to convert to a tensor.
    """
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
        inputs, batch_first=True
    )
    return outputs


class ConcatLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size: int, context_size: int):
        super(ConcatLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_dim = context_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.weight_mh = Parameter(torch.Tensor(4 * hidden_size, context_size))
        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        self._init_parameters()

    def forward(
        self, input: Tensor, date_contexts: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs: List[Tensor] = []
        pre_computed_md_layers = torch.mm(date_contexts, self.weight_mh.t())
        for i in range(len(inputs)):

            out, state = self._run_cell(
                inputs[i], pre_computed_md_layers, state
            )
            outputs += [out]
        return torch.stack(outputs), state

    def _run_cell(
        self, input: Tensor, md_layer: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + md_layer
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

    def _init_parameters(self):
        """ Basic randomization of parameters"""
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.xavier_normal_(weight)
            else:
                nn.init.zeros_(weight)  # bias vector


class ConcatLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size: int,
        context_size: int,
        num_layers: int,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [ConcatLSTMLayer(input_size, hidden_size, context_size)]
            + [
                ConcatLSTMLayer(hidden_size, hidden_size, context_size)
                for _ in range(num_layers - 1)
            ]
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        input: Tensor,
        date_contexts: Tensor,
        hx: Tuple[Tensor, Tensor] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states: List[Tuple[Tensor, Tensor]] = []
        output = input
        if self.batch_first:
            output = output.permute(1, 0, 2)

        states = hx
        if states is None:
            states = [
                (
                    torch.zeros(
                        output.shape[1], self.hidden_size, device=output.device
                    ),
                    torch.zeros(
                        output.shape[1], self.hidden_size, device=output.device
                    ),
                )
                for _ in range(self.num_layers)
            ]
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, date_contexts, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1

        if self.batch_first:
            output = output.permute(1, 0, 2)
        return output, output_states


class ConcatLSTMWarp(nn.Module):
    """This function implements a basic LSTM.

    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).

    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    input_shape : tuple
        The shape of an example input. Alternatively, use ``input_size``.
    input_size : int
        The size of the input. Alternatively, use ``input_shape``.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    dropout : float
        It is the dropout factor (must be between 0 and 1).

    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = LSTM(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        input_shape=None,
        input_size=None,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.reshape = False

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size.")

        # Computing the feature dimensionality
        if input_size is None:
            if len(input_shape) > 3:
                self.reshape = True
            input_size = torch.prod(torch.tensor(input_shape[2:])).item()

        self.rnn = ConcatLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            context_size=context_size,
        )

    def forward(self, x, date_contexts, hx=None, lengths=None):
        """Returns the output of the LSTM.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
            Starting hidden state.
        lengths : torch.Tensor
            Relative length of the input signals.
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # Flatten params for data parallel
        # self.rnn.flatten_parameters()

        # Pack sequence for proper RNN handling of padding
        if lengths is not None:
            x = pack_padded_sequence(x, lengths)

        # Support custom initial state
        if hx is not None:
            output, hn = self.rnn(x, date_contexts, hx=hx)
        else:
            output, hn = self.rnn(x, date_contexts)

        # Unpack the packed sequence
        if lengths is not None:
            output = pad_packed_sequence(output)

        return output, hn
