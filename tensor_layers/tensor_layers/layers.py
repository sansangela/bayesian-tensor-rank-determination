from ctypes import Union
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from . import low_rank_tensors
from .emb_utils import get_cum_prod,tensorized_lookup
import tensorly as tl


from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
# from ..common_types import _size_1_t, _size_2_t, _size_3_t


class TensorizedLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
    ):

        super(TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

    def forward(self, input, rank_update=True):

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        return F.linear(input,self.tensor.get_full().reshape([self.out_features,self.in_features]),self.bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class TensorizedEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False):

        super(TensorizedEmbedding,self).__init__()

        self.shape = shape
        self.tensor_type=tensor_type

        target_stddev = np.sqrt(2/(np.prod(self.shape[0])+np.prod(self.shape[1])))

        if self.tensor_type=='TensorTrainMatrix':
            tensor_shape = shape
        else:
            tensor_shape = self.shape[0]+self.shape[1]

        self.tensor = getattr(low_rank_tensors,self.tensor_type)(tensor_shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        self.parameters = self.tensor.parameters

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = np.prod(self.shape[0])
        self.emb_quant = np.prod(self.shape[1])

        self.padding_idx = padding_idx
        self.naive = naive

        self.cum_prod = get_cum_prod(shape)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()


    def forward(self, x,rank_update=True):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        #x_ind = self.ind2sub(x)

#        full = self.tensor.get_full()
#        full = torch.reshape(full,[self.voc_quant,self.emb_quant])
#        rows = full[x]
        if hasattr(self.tensor,"masks"):
            rows = tensorized_lookup(x,self.tensor.get_masked_factors(),self.cum_prod,self.shape,self.tensor_type)
        else:
            rows = tensorized_lookup(x,self.tensor.factors,self.cum_prod,self.shape,self.tensor_type)
#        rows = gather_rows(self.tensor, x_ind)

        rows = rows.view(x.shape[0], -1)

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        

        return rows.to(x.device)

#---------------------Define Scale Layer-------------------------------------------------------
class scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale, bit, half = False):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        if half:
            max_q = 2.0**(bit)-1.0
            min_q = 0
            quant = lambda x : fixed_point_quantize(x, wl=bit+1, fl=0, rounding="nearest")

        else:
            max_q = 2.0**(bit-1)-1.0
            min_q = -2.0**(bit-1)
            quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")
            # quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="stochastic")


        ctx.save_for_backward(input, scale)
        ctx.quant = quant
        ctx.input_div_scale = input/scale
        ctx.q_input = quant(ctx.input_div_scale)
        ctx.min_q = torch.tensor(min_q)
        ctx.max_q = torch.tensor(max_q)
        return scale * ctx.q_input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, scale= ctx.saved_tensors
        grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
        
        grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))

        grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q, max = ctx.max_q)

        return grad_input, grad_scale, None, None, None

class ScaleLayer(nn.Module):

   def __init__(self, scale=2**(-5), bit = 8, half = True):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([scale]))

       self.bit = bit
       self.half = True

    #    max_q = 2.0**(bit-1)-1.0
    #    min_q = -2.0**(bit-1)
    #    quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

    #    self.quant = quant
    #    self.min_q = min_q
    #    self.max_q = max_q

   def forward(self, input):
       return scale.apply(input,self.scale,self.bit,self.half)


class Quantized_Linear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                eta = None,
                device=None,
                dtype=None,
                bit = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5)
    ):

        super(Quantized_Linear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features

        self.bit = bit

        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

    def forward(self, input):
        self.weight = scale.apply(self.weight,self.scale_w,self.bit)
        self.bias = scale.apply(self.bias,self.scale_b,self.bit)
        
        return F.linear(input,self.weight,self.bias)



class Q_TensorizedLinear(nn.Linear):
    def __init__(self,
                in_features,
                out_features,
                bias=True,
                init=None,
                shape=None,
                tensor_type='TensorTrainMatrix',
                max_rank=20,
                em_stepsize=1.0,
                prior_type='log_uniform',
                eta = None,
                device=None,
                dtype=None,
                bit_w = 8,
                bit_b = 8,
                scale_w = 2**(-5),
                scale_b = 2**(-5)
    ):

        super(Q_TensorizedLinear,self).__init__(in_features,out_features,bias,device,dtype)

        self.in_features = in_features
        self.out_features = out_features
        target_stddev = np.sqrt(2/self.in_features)

        self.bit_w = bit_w
        self.bit_b = bit_b

        #shape taken care of at input time
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))


    def forward(self, input, rank_update=True):
        

        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
        Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
        

        output = input @ self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]).T
        output = scale.apply(output,self.scale_b,self.bit_b, False) + Q_bias

        self.Q_factors = Q_factors
        self.output = output

        return output
        
        # return F.linear(input,self.tensor.full_from_factors(Q_factors).reshape([self.out_features,self.in_features]),Q_bias)

    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()



class Q_conv2d_old(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 bit_w = 8,
                 bit_b = 8,
                 scale_w = 2**(-5),
                 scale_b = 2**(-5)
    ):
        super(Q_conv2d_old,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,device,dtype)

        self.stride = stride
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups

        self.bit_w = bit_w
        self.bit_b = bit_b
        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))
       

    def forward(self, input):
        Q_weight = scale.apply(self.weight,self.scale_w,self.bit_w)
        Q_bias = scale.apply(self.bias,self.scale_b,self.bit_b)
        
        output = F.conv2d(input,Q_weight,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        output = scale.apply(output,self.scale_b,self.bit_b)
        # print(output.shape)
        # print(Q_bias.shape)
        output = output.transpose(1,3)
        output = output + Q_bias
        output = output.transpose(1,3)

        return output


class Q_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 bit_w = 8,
                 bit_b = 8,
                 scale_w = 2**(-5),
                 scale_b = 2**(-5)
    ):
        super(Q_conv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups

        self.bit_w = bit_w
        self.bit_b = bit_b
        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init()
       
    def init(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        

    def forward(self, input):
        

        # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        Q_weight = scale.apply(self.weight,self.scale_w,self.bit_w, False)
        Q_bias = scale.apply(self.bias,self.scale_b,self.bit_b, False)
        output = F.conv2d(input,Q_weight,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        output = scale.apply(output,self.scale_b,self.bit_b,False)

        self.output = output

        output = output.transpose(1,3)
        output = output + Q_bias
        output = output.transpose(1,3)

        self.Q_weight = Q_weight

        return output

class Q_Tensorizedconv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = (3,3),
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrain',
                 max_rank=20,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 bit_w = 8,
                 bit_b = 8,
                 scale_w = 2**(-5),
                 scale_b = 2**(-5)
    ):
        super(Q_Tensorizedconv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups

        self.bit_w = bit_w
        self.bit_b = bit_b
        # self.max_q = 2.0**(bit-1)-1.0
        # self.min_q = -2.0**(bit-1)
        # self.quant = lambda x : fixed_point_quantize(x, wl=bit, fl=0, rounding="nearest")

        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))
        self.scale_b = nn.Parameter(torch.FloatTensor([scale_b]))

        # self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init()

        if shape == None:
            shape = self.get_tensor_shape(out_channels)
            shape = shape + self.get_tensor_shape(in_channels)
            shape = shape + list(kernel_size)


        target_stddev = 2/np.sqrt(self.in_channels*kernel_size[0]*kernel_size[1])
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

       
    def init(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def get_tensor_shape(self,n):
        if n==64:
            return [8,8]
        if n==128:
            return [8,16]
        if n==256:
            return [16,16]
        if n==512:
            return [16,32]

    def forward(self, input, rank_update = True):
        
        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False))
        Q_bias = (scale.apply(self.bias,self.scale_b,self.bit_b, False))
        # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        w = self.tensor.get_full_factors(Q_factors).reshape(self.out_channels,self.in_channels,*self.kernel_size)
        output = F.conv2d(input,w,bias = None, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        output = scale.apply(output,self.scale_b,self.bit_b,False)

        self.output = output

        output = output.transpose(1,3)
        output = output + Q_bias
        output = output.transpose(1,3)

        self.Q_weight = w

        return output


class Tensorizedconv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = (3,3),
                 stride= (1,1),
                 padding=(0,0),
                 dilation=(1,1),
                 groups=1,
                 bias = True,
                 padding_mode = 'zeros',
                 device=None,
                 dtype=None,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrain',
                 max_rank=20,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
    ):
        super(Tensorizedconv2d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        # self.output_padding = output_padding
        self.groups = groups

    

        # self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init()

        if shape == None:
            shape = self.get_tensor_shape(out_channels)
            shape = shape + self.get_tensor_shape(in_channels)
            shape = shape + list(kernel_size)


        target_stddev = 2/np.sqrt(self.in_channels*kernel_size[0]*kernel_size[1])
        self.tensor = getattr(low_rank_tensors,tensor_type)(shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

       
    def init(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def get_tensor_shape(self,n):
        if n==64:
            return [8,8]
        if n==128:
            return [8,16]
        if n==256:
            return [16,16]
        if n==512:
            return [16,32]

    def forward(self, input, rank_update = True):
        
        if self.training and rank_update:
            self.tensor.update_rank_parameters()
        
       
        # output = F.conv2d(input,self.weight,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

        w = self.tensor.get_full().reshape(self.out_channels,self.in_channels,*self.kernel_size)
        output = F.conv2d(input,w,bias = self.bias, stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)
        self.output = output
        return output


class Q_TensorizedEmbedding(nn.Module):
    def __init__(self,
                 init=None,
                 shape=None,
                 tensor_type='TensorTrainMatrix',
                 max_rank=16,
                 em_stepsize=1.0,
                 prior_type='log_uniform',
                 eta = None,
                 batch_dim_last=None,
                 padding_idx=None,
                 naive=False,
                 bit_w = 4,
                 scale_w = 2**(-5)):

        super(Q_TensorizedEmbedding,self).__init__()

        self.shape = shape
        self.tensor_type=tensor_type

        target_stddev = np.sqrt(2/(np.prod(self.shape[0])+np.prod(self.shape[1])))

        if self.tensor_type=='TensorTrainMatrix':
            tensor_shape = shape
        else:
            tensor_shape = self.shape[0]+self.shape[1]

        self.tensor = getattr(low_rank_tensors,self.tensor_type)(tensor_shape,prior_type=prior_type,em_stepsize=em_stepsize,max_rank=max_rank,initialization_method='nn',target_stddev=target_stddev,learned_scale=False,eta=eta)

        # Half Precision
        for i in range(len(self.tensor.factors)):
            self.tensor.factors[i] = self.tensor.factors[i].type(torch.float16)

        self.parameters = self.tensor.parameters

        self.batch_dim_last = batch_dim_last
        self.voc_size = int(np.prod(self.shape[0]))
        self.emb_size = int(np.prod(self.shape[1]))

        self.voc_quant = np.prod(self.shape[0])
        self.emb_quant = np.prod(self.shape[1])

        self.padding_idx = padding_idx
        self.naive = naive

        self.cum_prod = get_cum_prod(shape)

        self.bit_w = bit_w
        self.scale_w = nn.Parameter(torch.FloatTensor([scale_w]))



    def update_rank_parameters(self):
        self.tensor.update_rank_parameters()


    def forward(self, x,rank_update=True):

        xshape = list(x.shape)
        xshape_new = xshape + [self.emb_size, ]
        x = x.view(-1)

        Q_factors = []        
        for U in self.tensor.factors:
            Q_factors.append(scale.apply(U,self.scale_w,self.bit_w, False).type(torch.float16))     # Hard Code to half precision

        if hasattr(self.tensor,"masks"):
            rows = tensorized_lookup(x,self.tensor.get_masked_factors(),self.cum_prod,self.shape,self.tensor_type)
        else:
            # rows = tensorized_lookup(x,Q_factors,self.cum_prod,self.shape,self.tensor_type)

            # quantize intermediate tensors
            rows = tensorized_lookup(x,Q_factors,self.cum_prod,self.shape,self.tensor_type,scale,self.scale_w,self.bit_w)

        rows = rows.view(x.shape[0], -1)

        if self.padding_idx is not None:
            rows = torch.where(x.view(-1, 1) != self.padding_idx, rows, torch.zeros_like(rows))

        rows = rows.view(*xshape_new)

        if self.training and rank_update:
            self.tensor.update_rank_parameters()

        return rows.to(x.device)
