import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict

# for truncated normal
import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30., hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """
        Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!
        """
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


# test
# siren = Siren(2, 256, 3, 1, outermost_linear=True).cuda()
# dummy = torch.randn(5, 10, 2).cuda()
# dummy_out = siren(dummy)


# a simple MLP
class MLP(nn.Module):
    def __init__(self, in_features=2, out_features=1, n_neurons=256, n_layers=4, 
                       embedding_size=256, scale=0, dropout_rate=-1, hidden_act=None, output_act=None):
        """
        A simple MLP (can be used for NeRF papers)

        :param in_features: Input dimension of domain (2D, 3D)
        :param out_features: Output dimension of output field such as displacement field(2D, 3D)
        :param n_neurons: number of hidden neurons in hidden layers
        :param n_layers: number of hidden layers
        :param embedding_size: Embedding size of positional encoding
        :param scale: Scale value in positional encoding
        :param dropout_rate: Whether or not to use dropout (if 0 < ``dropout_rate`` < 1)
        :param hidden_act: activation function of hidden layers
        :param output_act: activation function of output layer
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.in_features = in_features
        self.n_neurons = n_neurons
        self.hooks_handles = []
        self.scale = scale
        self.B = torch.normal(0, 1, size=(self.embedding_size, self.in_features)) * self.scale
        if torch.cuda.is_available():
            self.B = self.B.cuda()
        
        self.net = []

        if hidden_act is None:
            hidden_act = nn.ReLU()

        for i in range(n_layers):
            if i == 0:
                self.net.append(nn.Linear(embedding_size * 2, n_neurons))
                self.net.append(hidden_act)
                if dropout_rate > 0:
                    self.net.append(nn.Dropout(p=dropout_rate))
            elif i == n_layers - 1:
                if output_act is None:
                    self.net.append(nn.Linear(n_neurons, out_features))
                else:
                    self.net.append(nn.Linear(n_neurons, out_features))
                    self.net.append(output_act)
            else:
                self.net.append(nn.Linear(n_neurons, n_neurons))
                self.net.append(hidden_act)
                if dropout_rate > 0:
                    self.net.append(nn.Dropout(p=dropout_rate))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.linear_orthogonal_initializer)

    def forward(self, coords):
        coords = (2. * math.pi * coords) @ self.B.T
        coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        coords = self.net(coords)
        return coords

    def input_encoder(self, x, a, b):
        return torch.cat([a * torch.sin((2. * math.pi * x) @ b.T), a * torch.cos((2. * math.pi * x) @ b.T)], dim=-1)

    def compute_new_posenc(self):
        bvals = 2. ** torch.linspace(0, self.scale, self.embedding_size//2) - 1.
        if torch.cuda.is_available():
            bvals = bvals.cuda()
        bvals = torch.stack([bvals, torch.zeros_like(bvals)], -1)
        bvals = torch.cat([bvals, torch.roll(bvals, 1, dims=-1)], 0) + 0
        avals = torch.ones((bvals.shape[0])) 
        if torch.cuda.is_available():
            avals = avals.cuda()
        return avals, bvals
    
    def compute_gaussian_enc(self):
        bvals = torch.normal(mean=0., std=1., size=(self.embedding_size, self.in_features))
        avals = torch.ones((bvals.shape[0])) 
        if torch.cuda.is_available():
            bvals = bvals.cuda()
            avals = avals.cuda()
        return avals, bvals

    def register_gated_activations(self, inputs, rate=0.5):
        """
        In case of Continual Learning with gated activations, we need to mask activations that are consistant for
          the entire of training using same task (resolution). This method register masks given percentage.
        
        :param inputs: Inputs to MLP
        :param rate: Percentage of zeroing out activations in each layer of MLP (float)
        """
        self.remove_forward_hooks()

        def change_act(mask):
            def hook_fn(module, inputs, activations):
                if module.training:
                    activations[mask] = 0
                return activations
            return hook_fn
        hooks_handles = []
        with torch.no_grad():
            x_proj = (2. * math.pi * inputs) @ self.B.T
            x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            for layer in self.net[:-1]:
                if isinstance(layer, torch.nn.modules.linear.Linear):
                    x_proj = layer(x_proj)
                    mask = torch.rand_like(x_proj) > rate
                    hooks_handles.append(layer.register_forward_hook(change_act(mask)))
        self.hooks_handles = hooks_handles

    def remove_forward_hooks(self):
        if len(self.hooks_handles) > 0:
            for h in self.hooks_handles:
                h.remove()
            self.hooks_handles = []

    def linear_orthogonal_initializer(self, m):
        """
        Orthogonal init for linear layers

        Please check:
        2. https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
        3. https://pytorch.org/docs/stable/nn.init.html

        :param m: Module m (rec: use module.apply(this method))
        """
        classname = m.__class__.__name__
        gain = 1.0 * np.sqrt(max(self.n_neurons / self.embedding_size, 1))
        if classname.find('Linear') != -1:
            torch.nn.init.orthogonal_(m.weight, gain=gain)  # gain is the same in tf and pt
            torch.nn.init.constant_(m.bias, 0.0)

# test
# model = MLP(1, 1)
# dummy_in = torch.randn(1, 10, 1)
# dummy_out, coords = model(dummy_in)
# print(dummy_out.shape)

class MultiHeadedMLP(nn.Module):
    def __init__(self, in_features=2, out_features=1, n_neurons=256, n_layers=4, embedding_size=256, n_heads=10):
        """
        A Multi Headed MLP (can be used for Continual Learning)

        :param in_features: Input dimension of domain (2D, 3D)
        :param out_features: Output dimension of output field such as displacement field(2D, 3D)
        :param n_neurons: number of hidden neurons in hidden layers
        :param n_layers: number of hidden layers
        :param embedding_size: Embedding size of positional encoding
        :param n_heads: Number of heads for final output (specificly for continal learning)
        :param output_act: activation function of output layer
        """
        super().__init__()

        self.n_neurons = n_neurons
        self.embedding_size = embedding_size
        self.old_scale = 1.

        self.base_model = MLP(in_features=in_features, out_features=out_features, n_neurons=n_neurons,
                              n_layers=n_layers, embedding_size=embedding_size, scale=self.old_scale,
                              hidden_act=nn.ReLU())
        self.base_model.net = self.base_model.net[:-1]

        # for each we create a linear to number of desired outputs
        heads = []
        for i in range(n_heads):
            heads.append(nn.Linear(in_features=n_neurons, out_features=out_features))
        self.heads = nn.ModuleList(heads)
        self.heads.apply(self.linear_orthogonal_initializer)

    def forward(self, coords, head_idx):
        out = self.base_model(coords)
        out = self.heads[head_idx](out)
        return out

    def change_scale_value(self, scale):
        """
        Changes the ``scale`` value of Fourier feature embedding. 
        Will be used to update ``scale`` as the task value for each head.

        :param scale: New scale for Fourier feature embeddings in NeRF based MLP 
        """
        self.base_model.B = self.base_model.B / self.old_scale * scale
        self.old_scale = scale

    def linear_orthogonal_initializer(self, m):
        """
        Orthogonal init for linear layers

        Please check:
        2. https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
        3. https://pytorch.org/docs/stable/nn.init.html

        :param m: Module m (rec: use module.apply(this method))
        """
        classname = m.__class__.__name__
        gain = 1.0 * np.sqrt(max(self.n_neurons / self.embedding_size, 1))
        if classname.find('Linear') != -1:
            torch.nn.init.orthogonal_(m.weight, gain=gain)  # gain is the same in tf and pt
            torch.nn.init.constant_(m.bias, 0.0)


# try to follow pytorch pattern instead of Julia
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
class Generator(nn.Module):
    def __init__(self, in_features, design, hidden_act=None, output_act=None, init_weights=None):
        """
        A Generator based on GAN approaches

        :param in_features: Number of random seed generated numbers
        :param design: Shape of final design variable (2D)
        :param hidden_act: activation function of hidden layers
        :param output_act: activation function of output layer
        :param init_weights: weight initialization method (default is ``__weight_init`` in case of ``None``)
        """
        super().__init__()

        self.in_features = in_features
        self.design = design
        if hidden_act is None:
            hidden_act = nn.Tanh()
        self.hidden_act = hidden_act
        self.output_act = output_act

        # currently works for 180x60 final design
        self.linear = nn.Linear(in_features, 4)
        self.conv_transpose1 = nn.ConvTranspose2d(4, 2, kernel_size=7, padding=2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(2, 1, kernel_size=4, padding=2, stride=2)

        if init_weights is None:
            self.apply(self.__init_weights)

        # a simple average filtering
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(torch.ones_like(self.conv1.weight) * (1/9))
            self.conv1.bias = nn.Parameter(torch.tensor([0.]))
            self.conv1.requires_grad = False
        
        

    def forward(self, coords):
        output = self.linear(coords)
        output = output.view(output.shape[0], output.shape[-1], self.design[0] // 4, self.design[1] // 4)
        output = self.conv_transpose1(output)
        output = self.hidden_act(self.conv_transpose2(output))
        output = self.conv1(output)
        return output

    @staticmethod
    def __init_weights(m):
        classname = m.__class__.__name__
        if classname.find('ConvTranspose2d') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.zeros_(m.bias)


# test
# design = np.array([200, 100])
# latent = (design[0]//4) * (design[1]//4)
# dummy_in = torch.randn(1, latent, 1)
# generator = Generator(1, design=design)
# dummy_out = generator(dummy_in)
# print(dummy_out.shape)


# Google's Generator
# https://github.com/google-research/neural-structural-optimization/blob/master/neural_structural_optimization/models.py
class CNNModel(nn.Module):
    def __init__(self,
                 gridDimension,
                 latent_size=128, 
                 dense_channels=32,
                 resizes=(1, 2, 2, 2, 1),
                 conv_filters=(128, 64, 32, 16, 1),
                 offset_scale=10,
                 kernel_size=(5, 5),
                 dense_init_scale=1.0,
                 activation=nn.Tanh(),
                 ):
        """
        A generator based model which input is part of the parameters and generates a 2D single channel tensor out of latent
        vector beta as its input.
        
        :param gridDimension: Target output shape
        :param latent_size: Input beta shape
        :param resizes: Amount of rescale sizes for upsampling using F.interpolate
        :param conv_filters: Number of output channels for each convlutional layer
        :param offset_scale: Scale hyperparameter for ``AddOffset`` layers
        :param kernel_size: Kernel size of convolutional layers
        :param dense_init_scale: Scale for initialization of linear layers
        :param activation: Activation function of intermediate layers

        :return: A 2D tensor of (batch, 1, gridDimension[0], gridDimension[1])
        """

        super().__init__()


        if len(resizes) != len(conv_filters):
            raise ValueError('resizes and filters must be same size')

        self.dense_channels = dense_channels
        self.activation = activation
        self.resizes = resizes
        self.conv_filters = conv_filters
        self.offset_scale = offset_scale
        self.dense_init_scale = dense_init_scale
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        
        total_resize = int(np.prod(resizes))
        self.h = gridDimension[0] // total_resize
        self.w = gridDimension[1] // total_resize

        n_filters = self.h * self.w * dense_channels
        self.n_filters = n_filters

        self.linear = nn.Linear(latent_size, n_filters)
        self.linear.apply(self.linear_orthogonal_initializer)

        self.conv = []
        self.conv.append(nn.Conv2d(dense_channels, self.conv_filters[0], kernel_size))
        for i in range(1, len(self.conv_filters)):    
            self.conv.append(nn.Conv2d(self.conv_filters[i-1], self.conv_filters[i], kernel_size))
        self.conv = nn.ModuleList(self.conv)
        self.conv.apply(self.variance_scaling_initializer)

        self.add_offset = []
        self.compute_offset_bias_shapes()
        for i in range(len(self.offset_bias_shapes)):
            self.add_offset.append(AddOffset(input_shape=self.offset_bias_shapes[i], scale=self.offset_scale))
        self.add_offset = nn.ModuleList(self.add_offset)

    def forward(self, data):
        # data = data.clone().detach().requires_grad_(True)  # enable grad computation wrt input (eg make input trainable)
        data = self.linear(data)
        data = data.view(data.shape[0], self.dense_channels, self.h, self.w)

        for i, resize in enumerate(self.resizes):
            data = self.activation(data)
            data = F.interpolate(input=data, scale_factor=resize, mode='bilinear', align_corners=False)
            data = self.global_normalization(data)
            data = F.pad(data, pad=self.compute_same_padding(data.shape[-2:], self.kernel_size))
            data = self.conv[i](data)

            if self.offset_scale != 0:
                data = self.add_offset[i](data)
        data = data.squeeze(dim=0)
        return data


    def compute_same_padding(self, data_shape, kernel_shape=(5, 5)):
        """
        Computes amount of padding needed in each side of a 2D tensor based on tensorflow semantics

        :param data_shape: Shape of input data to be padded
        :param kernel_shape: Shape of 2D kernel for the given convolution layer over input data with shape ``data_shape``
        :return: Tuple of (left, right, top, bot) of ints as input args to ``F.pad()``
        """

        in_height, in_width = data_shape
        filter_height, filter_width = kernel_shape
        strides=(None,1,1)
        out_height = np.ceil(float(in_height) / float(strides[1]))
        out_width  = np.ceil(float(in_width) / float(strides[2]))

        #The total padding applied along the height and width is computed as:
        if (in_height % strides[1] == 0):
            pad_along_height = max(filter_height - strides[1], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[1]), 0)
        if (in_width % strides[2] == 0):
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)
        
        # left, right, top, bottom for F.pad
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return (pad_left, pad_right, pad_top, pad_bottom)


    def compute_offset_bias_shapes(self):
        """
        Computes input shapes to be able to construct parameter bias with same shape of input to an ``AddOffset`` layer 
          corresponding to ``self.conv`` layers

        Run this method before using model itself to first populate necessary layers with correct shapes or
          manually pass shapes by setting ``self.offset_bias_shapes`` which would be a complicated task manually.
        
        :param data: Input to the model
        """

        dummy_data = torch.randn(1, self.latent_size, )  # assume batch_sizes=1 
        self.offset_bias_shapes = []

        with torch.no_grad():
            dummy_data = self.linear(dummy_data)
            dummy_data = dummy_data.view(1, self.dense_channels, self.h, self.w)
            for i in range(len(self.conv)):
                dummy_data = F.interpolate(input=dummy_data, scale_factor=self.resizes[i], mode='bilinear', align_corners=False)
                dummy_data = F.pad(dummy_data, pad=self.compute_same_padding(dummy_data.shape[-2:], self.kernel_size))
                dummy_data = self.conv[i](dummy_data)
                self.offset_bias_shapes.append(dummy_data.shape)


    def global_normalization(self, data, epsilon=1e-6):
        """
        Computes normalization over global tensor (rather than batch, e.g. dim=[1, 2, 3] in case of 4D tensor)

        :param epsilon: To prevent ``nan`` in ``torch.rsqrt``
        :return: Normalized input data
        """

        mean, variance = self.moments(data, dim=list(range(len(data.shape)))[1:])
        data = data - mean
        data = data * torch.rsqrt(variance + epsilon)
        return data


    @staticmethod
    def moments(data, dim):
        """
        Computes mean and then variance from obtained mean

        TF equivalent: tf.nn.moments

        :param dim: Reduce over dimensions (helps to achieve global normalization)
        :return: Mean and Variance of input data passed to ``global_normalization``
        """
        mean = torch.mean(data, dim=dim)
        # var = torch.sqrt((torch.sum((data-mean)**2))/torch.prod(torch.tensor(data.shape).float()))
        var = torch.var(data, dim=dim)
        return mean, var


    @staticmethod
    def variance_scaling_initializer(m):
        """
        Kaiming He init for conv2d layers (TF implementation is not He even though they claim it is, see ref #4)

        Please check:
        1. https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
        2. https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling
        3. https://pytorch.org/docs/stable/nn.init.html
        4. https://stats.stackexchange.com/questions/484062/he-normal-keras-is-truncated-when-kaiming-normal-pytorch-is-not

        :param m: Module m (rec: use module.apply(this method))
        """  
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.kaiming_normal_(m.weight, 0.0, nonlinearity='sigmoid')  # TODO: error in scale of 5e4
            torch.nn.init.constant_(m.bias, 0.0)


    def linear_orthogonal_initializer(self, m):
        """
        Orthogonal init for linear layers

        Please check:
        2. https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
        3. https://pytorch.org/docs/stable/nn.init.html

        :param m: Module m (rec: use module.apply(this method))
        """
        classname = m.__class__.__name__
        gain = self.dense_init_scale * np.sqrt(max(self.n_filters / self.latent_size, 1))
        if classname.find('Linear') != -1:
            torch.nn.init.orthogonal_(m.weight, gain=gain)  # gain is the same in tf and pt
            torch.nn.init.constant_(m.bias, 0.0)


class AddOffset(nn.Module):
    def __init__(self, input_shape, scale=1):
        """
        Offset layer which learns a weight with same shape of input which is scaled by a constant float ``scale``

        :param input_shape: Input shape to creates a weight matrix as the parameter with the same shape as input
        :param scale: Constant scale factor as the weight of parameter
        """

        super().__init__()
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(input_shape), requires_grad=True)

    def forward(self, data):
        return data + self.scale * self.bias


# clonned from https://github.com/toshas/torch_truncnorm
# TODO include his work in acknoledgement and references of code
CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, a, b, eps=1e-8, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        self._dtype_min_gt_0 = torch.tensor(torch.finfo(self.a.dtype).eps, dtype=self.a.dtype)
        self._dtype_max_lt_1 = torch.tensor(1 - torch.finfo(self.a.dtype).eps, dtype=self.a.dtype)
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def expand(self, batch_shape, _instance=None):
        # TODO: it is likely that keeping temporary variables in private attributes violates the logic of this method
        raise NotImplementedError


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'a': constraints.real,
        'b': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, a, b, eps=1e-8, validate_args=None):
        self.loc, self.scale, self.a, self.b = broadcast_all(loc, scale, a, b)
        a_standard = (a - self.loc) / self.scale
        b_standard = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a_standard, b_standard, eps=eps, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale
