import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        output_width = A.shape[2] - self.kernel + 1
        output_height = A.shape[3] - self.kernel + 1

        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        self.maxIndex = np.zeros((A.shape[0], A.shape[1], output_width, output_height, 2), dtype=int)
        
        for i in range(output_width):

            for j in range(output_height):

                input_slice = A[:,:,i:i+self.kernel,j:j+self.kernel]

                Z[:,:,i,j] = np.max(input_slice, axis=(2, 3))

                max_index_flat = np.argmax(input_slice.reshape(A.shape[0], A.shape[1], -1), axis=-1)

                row_index, col_index = np.unravel_index(max_index_flat, (self.kernel, self.kernel))

                self.maxIndex[:, :, i, j, 0] = row_index
                self.maxIndex[:, :, i, j, 1] = col_index

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)

        for i in range(self.maxIndex.shape[2]):

            for j in range(self.maxIndex.shape[3]):

                row_index = self.maxIndex[:, :, i, j, 0]

                col_index = self.maxIndex[:, :, i, j, 1]

                batch_indices = np.arange(self.A.shape[0])[:, None]
                channel_indices = np.arange(self.A.shape[1])

                dLdA[batch_indices, channel_indices, i + row_index, j + col_index] += dLdZ[:, :, i, j]
        
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        output_width = A.shape[2] - self.kernel + 1
        output_height = A.shape[3] - self.kernel + 1

        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))

        for i in range(output_width):

            for j in range(output_height):

                input_slice = A[:,:,i:i+self.kernel,j:j+self.kernel]

                Z[:,:,i,j] = np.mean(input_slice, axis=(2,3))
                

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros_like(self.A)

        for i in range(dLdZ.shape[2]):

            for j in range(dLdZ.shape[3]):

                batches = np.arange(self.A.shape[0])[:, None]
                channels = np.arange(self.A.shape[1])

                dLdA[batches, channels, i:i+self.kernel, j:j+self.kernel] += dLdZ[:,:,i,j][:, :, None, None] / (self.kernel ** 2)


        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d =Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        delta_out = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(delta_out)
        
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        delta_out = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(delta_out)
        
        return dLdA
