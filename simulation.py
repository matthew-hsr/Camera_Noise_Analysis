import numpy as np
import math
import random
import time
import pickle
import scipy.signal
import numpy

class simulation_environment():
    """Define the environment for simulation."""

    def __init__(
        self, length_of_chip, read_noise, 
        add_shot_noise_flag, add_EM_gain_noise, 
        exposure_time, binning_sequence, num_of_samples,
        normalization, target_size, num_of_disks, disk_intensity):

        self.length_of_chip = length_of_chip
        self.read_noise = read_noise
        self.add_shot_noise_flag = add_shot_noise_flag # Boolean
        self.add_EM_gain_noise = add_EM_gain_noise # Boolean
        self.exposure_time = exposure_time
        self.binning_sequence = binning_sequence
        self.num_of_samples = num_of_samples
        self.normalization = normalization
        self.target_size = target_size # 8 or 16?
        self.num_of_disks = num_of_disks
        self.disk_intensity = disk_intensity
        assert len(exposure_time) == len(binning_sequence), 'Error! \nExposure time and binning sequence have different length.'
        assert len(exposure_time[0]) == len(binning_sequence), 'Error! \nExposure time and binning sequence have different length.'

    def generate_simulation_data(self):
        self.generate_GT_2D_disks()
        self.binning_2D()

        if self.add_shot_noise_flag:
            self.add_shot_noise()
        self.add_read_noise()
        self.convolve_GT()
        self.split_binned_data()
        self.get_single_pixel_bin_data()
        self.single_pixel_bin_data

    def generate_GT_2D(self):
        """Generate GT for 2d samples."""
        self.GT = np.random.rand(
            self.num_of_samples, self.length_of_chip, self.length_of_chip)
        for index in range(len(self.GT)):
            self.GT[index, :, :] = self.GT[index, :, :] / np.sum(np.sum(self.GT[index, :, :])) * self.normalization 
        # Normalize each sample to a certain value?

    def generate_GT_2D_disks(self, disks_width = None):
        """Generate GT for 2d samples."""
        length_of_chip = self.length_of_chip
        disks_width = self.target_size
        intensity = self.disk_intensity
        num_of_disks = self.num_of_disks
        self.GT = np.zeros((self.num_of_samples, self.length_of_chip, self.length_of_chip))
        for sample_index in range(self.num_of_samples):
            disks_positions = np.random.randint(0, self.length_of_chip, size = (num_of_disks, 2))
            X, Y = np.meshgrid(np.arange(0,length_of_chip), np.arange(0,length_of_chip))
            for disk_index in range(num_of_disks):
                self.GT[sample_index, :, :]+= intensity * np.exp(- ((X - disks_positions[disk_index,0])**2 + (Y - disks_positions[disk_index,1])**2)/ (2 * disks_width**2)) / np.sqrt(2*np.pi*disks_width**2)
        self.GT = self.GT.astype(int)

    def bin_2D(self, image, binning_X, binning_Y, exposure_time):
        return scipy.signal.convolve(
            image, np.ones((1,binning_Y,binning_X)), mode='valid', method='direct')[:,::binning_Y, ::binning_X] * exposure_time

    def binning_2D(self):
        """Create a 2-d array of objects, channels last."""
        self.data_list=[]
        for index_Y, binning_Y in enumerate(self.binning_sequence):
            for index_X, binning_X in enumerate(self.binning_sequence):
                self.data_list.append(
                    self.bin_2D(self.GT, binning_X, binning_Y, self.exposure_time[index_X, index_Y]))
                # print(np.min(np.min(np.min(self.data_list[-1]))))

    def add_shot_noise(self):
        """Add shot noise to the data"""
        for index, data in enumerate(self.data_list):
            self.data_list[index] = np.random.poisson(data)

    def add_read_noise(self):
        """Add zero centered Gaussian read noise to the GT"""
        for index, data in enumerate(self.data_list):
            self.data_list[index] = (data + 
            np.random.normal(loc = 0, scale=self.read_noise, size=data.shape))

    def split_binned_data(self):
        """Get train_X by putting the binned data into their separate pixels"""
        self.train_X = np.zeros((self.num_of_samples, self.length_of_chip, self.length_of_chip, len(self.binning_sequence)**2))
        index=0
        for index_Y, binning_Y in enumerate(self.binning_sequence):
            for index_X, binning_X in enumerate(self.binning_sequence):
                self.train_X[:, :, :, index] = np.tile(self.data_list[index], [binning_Y, binning_X])
                index+=1

    def convolve_GT(self):
        """Get the convolution of ground truth, to be our training target"""
        self.train_y = scipy.signal.convolve(
            self.GT, np.ones((1,self.target_size,self.target_size)),mode='valid')

    def get_single_pixel_bin_data(self):
        """Get the single pixel results, for comparison"""
        self.single_pixel_bin_data = (
            np.random.poisson(
                self.GT * np.sum(np.sum(self.exposure_time)))
                + np.random.normal(loc = 0, scale=self.read_noise, size=self.GT.shape))
