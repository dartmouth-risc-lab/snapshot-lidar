import numpy as np
import cv2
from skimage.io import imread, imshow
from scipy.constants import c, pi
from skimage.color import rgb2gray
from utils import compute_I_analytical
from pathlib import Path

'''
    Base class for simulating an image from given per pixel intensity and depth.
    
    Assumes images are grayscale np arrays.

    intensity_image: GRAYSCALE ground truth intensity image
    phase_image: GRAYSCALE ground truth depth image
    omega: angular modulation frequency of the signal (convert Hz to rads)
    illum_intensity_scale: intensity of the illumination signal
    rotation_angle: in degrees
'''


class Simulation:
    def __init__(self, intensity_image, depth_image, omega=2*pi*1e6,  illum_intensity_scale=1, rotation_angle=0):
        
        self.using_real_data = False
        self.Gaussian = False
        self.sigma = 0
        self.using_quad = False
        self.rotation_angle=rotation_angle
    
        self.intensity_gt = intensity_image
        self.depth_gt = depth_image
        
        self.height, self.width = self.intensity_gt.shape
        
        if rotation_angle != 0:
            self.rotation_M = cv2.getRotationMatrix2D(((self.width-1)/2.0,(self.height-1)/2.0),rotation_angle,1)
            self.intensity_gt = cv2.warpAffine(self.intensity_gt, self.rotation_M, (self.width, self.height))
            self.depth_gt = cv2.warpAffine(self.depth_gt, self.rotation_M, (self.width, self.height))

        self.omega = omega
        self.T = 2.0 * pi / omega
        self.illum_intensity_scale = illum_intensity_scale

        self.phase_gt = (2.0 * self.depth_gt * omega )/ c
        self.A = self.illum_intensity_scale * self.intensity_gt
       
    """
       Synthesize quadrature measurements from the ground truth intensity

    """    
    def create_quad_intensity(self, method=''):
        self.I0 = compute_I_analytical(self.A, 0, self.phase_gt, method)
        self.I90 = compute_I_analytical(self.A, pi/2.0, self.phase_gt, method)
        self.I180 = compute_I_analytical(self.A, pi, self.phase_gt, method)
        self.I270 = compute_I_analytical(self.A, pi * 3.0 / 2.0, self.phase_gt, method)          

        if self.rotation_angle != 0:
            self.I0 = cv2.warpAffine(self.I0, self.rotation_M,(self.width,self.height))
            self.I90 = cv2.warpAffine(self.I90, self.rotation_M,(self.width,self.height))
            self.I180 = cv2.warpAffine(self.I180, self.rotation_M,(self.width,self.height))
            self.I270 = cv2.warpAffine(self.I270, self.rotation_M,(self.width,self.height))

        self.quad_images = [self.I0, self.I90, self.I180, self.I270]

    """
       Create quadrature measurements from real world captures. 
       
       images - list of np arrays representing the captured quadrature images
    """
    def create_quad_intensity_from_data(self, images=[]):
        
        self.using_quad = True
        self.using_real_data = True
        
        self.I0 = images[0]
        self.I90 = images[1]
        self.I180 = images[2]
        self.I270 = images[3]
    
        if self.rotation_angle != 0:
            self.I0 = cv2.warpAffine(self.I0, self.rotation_M,(self.width,self.height))
            self.I90 = cv2.warpAffine(self.I90, self.rotation_M,(self.width,self.height))
            self.I180 = cv2.warpAffine(self.I180, self.rotation_M,(self.width,self.height))
            self.I270 = cv2.warpAffine(self.I270, self.rotation_M,(self.width,self.height))
        
        fpn = (self.I0 + self.I90 + self.I180 + self.I270) / 4.0
        self.I0 = self.I0 - fpn
        self.I90 = self.I90 - fpn
        self.I180 = self.I180 - fpn
        self.I270 = self.I270 - fpn
        
        self.quad_images = [self.I0, self.I90, self.I180, self.I270]
 
    """
       Compute phase using the conventional four-bucket method. 
       
    """
    def compute_phase_conventional(self, images=[]):
        if images == []:
            self.phase_from_data = np.arctan2(self.I90 - self.I270, self.I0 - self.I180)
            return self.phase_from_data
        else:
            self.phase_from_data = np.arctan2(images[1] - images[3], images[0] - images[2])
            return self.phase_from_data
    
    """
       Compute amplitude using the conventional four-bucket method. 
       
    """
    def compute_amplitude_conventional(self, images=[]):
        if images == []:
            return np.sqrt((self.I90 - self.I270)**2 + (self.I0 - self.I180)**2) / 2
        else:
            return np.sqrt((images[1] - images[3])**2 + (images[0] - images[2])**2) / 2
    
    """
       Compute and store the fft image. 
       
    """
    def calculate_fft(self):
        self.img_32 = np.float32(self.intensity_gt) # make sure image is float 32

        self.fft = np.fft.fftshift(np.fft.fft2(self.img_32))  # Fourier transform of the image
        self.fft_reverse = np.fft.ifft2(np.fft.ifftshift(self.fft))
    
    """
       Display the metadata associated with this simulation.
       
    """
    def get_metadata(self):
        print("image width, height: ", self.width, self.height)
        print("using Gaussian filtering? : ", self.Gaussian)
        if self.Gaussian:
            print("sigma = ", self.sigma)
        print("using real world data? : ", self.using_real_data)
        print("using quad data to calculate phase/amp?", self.using_quad)
