import torch
import torch.nn as nn
from torch.autograd import Variable


# Based on: https://github.com/yiweilu3/CONV-VRNN-for-Anomaly-Detection

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, use_bn):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        use_bn: bool
            Whether or not to use BatchNormalization after each layer.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                         out_channels=4 * self.hidden_dim,
                         kernel_size=self.kernel_size,
                         padding=self.padding,
                         bias=self.bias)

        if not use_bn:
            self.conv = conv
        else:
            self.conv = nn.Sequential(conv, nn.BatchNorm2d(conv.out_channels))

    def forward(self, input_tensor, cur_state, return_gates=False):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        if return_gates:
            return h_next, c_next, torch.stack((i, f, o, g), dim=1)
        return h_next, c_next

    def init_hidden(self, batch_size):
        v1 = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        v2 = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        if torch.cuda.is_available():
            v1 = v1.cuda()
            v2 = v2.cuda()
        return v1, v2


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, use_bn=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.use_bn = use_bn

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            ks = self.kernel_size[i]
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=ks,
                                          bias=self.bias,
                                          use_bn=self.use_bn))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state, return_gates=False):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            4-D Tensor either of shape (t, c, h, w) 
            
        Returns
        -------
        last_state_list, layer_output
        """

        cur_layer_input = input_tensor

        all_gates = []
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            res = self.cell_list[layer_idx](cur_layer_input, cur_state=[h, c], return_gates=return_gates)
            if return_gates:
                h, c, gates = res
                all_gates.append(gates)
            else:
                h, c = res
            hidden_state[layer_idx] = h, c
            cur_layer_input = h

        if return_gates:
            return h, hidden_state, all_gates
        return h, hidden_state

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
