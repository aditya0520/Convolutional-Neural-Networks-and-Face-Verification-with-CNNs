import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        self.output_height = self.A.shape[2] - self.kernel_size + 1
        self.output_width =  self.A.shape[3] - self.kernel_size + 1

        Z = np.zeros((A.shape[0], self.out_channels, self.output_height, self.output_width))

        for i in range(self.output_height):

            for j in range(self.output_width):

                input_slice = self.A[:, :, i:i+self.kernel_size, j:j+self.kernel_size]

                Z[:, :, i, j] = np.tensordot(input_slice, self.W, axes=([1,2,3],[1,2,3]))
     
        return Z + self.b[None,:, None, None]

        

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        input_height = self.A.shape[2]
        input_width = self.A.shape[3]
        dLdA = np.zeros_like(self.A)

        self.dLdb = np.sum(dLdZ, axis=(0,2,3))

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):

                input_slice = self.A[:, :, i:i+self.output_height, j:j+self.output_width]

                self.dLdW[:, :, i, j] = np.tensordot(dLdZ, input_slice, axes=([0,2,3],[0,2,3]))


        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0),
                                    (self.kernel_size - 1, self.kernel_size - 1), 
                                    (self.kernel_size - 1, self.kernel_size - 1) ), mode='constant')

        flipped_W = np.flip(self.W, axis=(3, 2))

        for i in range(input_height):

            for j in range(input_width):

                padded_dLdZ_slice = padded_dLdZ[:, :, i:i+self.kernel_size, j:j+self.kernel_size]

                dLdA[:,:,i,j] = np.tensordot(padded_dLdZ_slice, flipped_W, axes=([1,2,3],[0,2,3]))


        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels,out_channels,kernel_size,weight_init_fn,bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        Z = np.pad(A, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)))

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(Z)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        if self.pad > 0:
            dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
