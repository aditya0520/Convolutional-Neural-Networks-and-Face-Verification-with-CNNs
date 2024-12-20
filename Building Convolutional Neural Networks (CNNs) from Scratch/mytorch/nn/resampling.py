import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.A = A
        Z = np.zeros((A.shape[0], A.shape[1], self.upsampling_factor * (A.shape[2] - 1) + 1))

        Z[:,:,::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = np.zeros_like(self.A)
        dLdA = dLdZ[:,:,::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        self.A = A
        Z = np.zeros((A.shape[0], A.shape[1], (A.shape[2] - 1) // self.downsampling_factor + 1))

        Z = A[:,:,::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = np.zeros_like(self.A)
        dLdA[:,:,::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.A = A
        output_height = self.upsampling_factor * (A.shape[2] - 1) + 1
        output_width =  self.upsampling_factor * (A.shape[3] - 1) + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))

        Z[:,:,::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros_like(self.A)
        dLdA = dLdZ[:,:,::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        self.A = A
        output_height = (A.shape[2] - 1) // self.downsampling_factor + 1
        output_width = (A.shape[3] - 1) // self.downsampling_factor + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_height, output_width))

        Z = A[:,:,::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = np.zeros_like(self.A)
        dLdA[:,:,::self.downsampling_factor, ::self.downsampling_factor] = dLdZ
        
        return dLdA