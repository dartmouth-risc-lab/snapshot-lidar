import numpy as np
from skimage.io import imread, imshow
from math import floor
from scipy.ndimage import gaussian_filter, gaussian_filter1d, shift
from scipy.constants import c, pi
from skimage.color import rgb2gray
from utils import compute_I_analytical, create_mask
from simulation import Simulation
import scipy.interpolate as interp


'''
    Child class of Simulation for simulating a fft shift composite image from given per pixel intensity and depth.

    intensity_image: path to the ground truth intensity image
    depth_image: path to the ground truth depth image
    omega: angular modulation frequency of the signal (convert Hz to rads)
    illum_intensity_scale: intensity of the illumination signal
  
'''

class FFT_Shift(Simulation):
    
    def __init__(self, k, intensity_image, depth_image, omega=2*pi*1e6,  illum_intensity_scale=1, rotation_angle=0):
        self.k = k
        super().__init__(intensity_image, depth_image, omega=omega, illum_intensity_scale=illum_intensity_scale, rotation_angle=rotation_angle)


    """
        Generate the composite image given k, orientation, and the ground truth intensity and phase info
        
        If use_Gaussian is true, process the intensity with a low-pass Gaussian filter with sigma specified
        
        Currently only supports y or x spatially varying theta
        
        phase_variation_axis: 'x'|'y'
        use_Gaussian: dimension of the gaussian kernel, '1d'|'2d'|None
        sigma: standard deviation of the gaussian kernel

    """
    def create_composite_intensity(self, phase_variation_axis='x', use_Gaussian='1d', sigma=0, method=''):
        
        A = self.illum_intensity_scale * np.abs(self.intensity_gt)
            
        # illumination theta
        if phase_variation_axis == 'x':
            tmp_x = np.linspace(0, self.width - 1, num=self.width)
            tmp_y = np.arange(0, self.height)
            theta, _ = np.meshgrid(tmp_x, tmp_y)  # theta(x) = ky, this is the illumination varied by y
            
        elif phase_variation_axis == 'y':
            tmp_x = np.arange(0, self.width)
            tmp_y = np.linspace(0, self.height - 1, num=self.height)
            _, theta = np.meshgrid(tmp_x, tmp_y)  # theta(x) = kx, this is the illumination varied by x

        else:
            raise Exception("Only x or y phase variation axes are currently supported")
            
        theta *= self.k
            
        # Kill overlapping high frequency from ground truth intensity
        if use_Gaussian == '2d' and sigma != 0:
            self.Gaussian = True
            self.sigma = sigma
            F_tmp = gaussian_filter(A*(np.cos(self.phase_gt) + 1j*np.sin(self.phase_gt)), sigma, mode='constant')
            A = abs(F_tmp)
            phi = np.angle(F_tmp)
        elif use_Gaussian == '1d' and sigma != 0:
            self.Gaussian = True
            self.sigma = sigma
            filter_axis = 1 if phase_variation_axis == 'x' else 0
            F_tmp = gaussian_filter1d(A*(np.cos(self.phase_gt) + 1j*np.sin(self.phase_gt)), sigma, axis=filter_axis, mode='constant')
            A = abs(F_tmp)
            phi = np.angle(F_tmp)
        else:
            phi = self.phase_gt

        self.composite_image = compute_I_analytical(A, theta, phi, method)
        self.F_filtered = np.fft.fftshift(np.fft.fft2(self.composite_image))
    
    """
        Initialize the composite image from a captured snapshot image
    """        
    def create_composite_intensity_snapshot(self, image):
        self.composite_image = image
        
    
    """
        Compute a composite image with spatially varying phase shift theta=kx from real world data
        Can give images other than the quadrature measurements, but defaults to quadrature
        
        images: list of captured images
        num_phase_shifts: number of measurements used to create the composite image
        phase_variation_axis: 'x'|'y'
        use_Gaussian: dimension of the gaussian kernel, '1d'|'2d'|None
        sigma: standard deviation of the gaussian kernel
        
    """
    def create_composite_intensity_from_data(self, images=[], num_phase_shifts=4, phase_variation_axis='x', use_Gaussian='1d', sigma=0):
        
        self.using_real_data = True
        if images == []:
            images = self.quad_images
        
        filtered_images = []
        # create and filter the hologram
        if use_Gaussian == '2d' and sigma != 0:
            self.Gaussian = True
            self.sigma = sigma
            A = super().compute_amplitude_conventional(images)
            phi = super().compute_phase_conventional(images)
            
            F_tmp = gaussian_filter(A*(np.cos(phi) + 1j*np.sin(phi)), sigma, mode='constant') # gaussian filter the hologram
            A_cur = 2*abs(F_tmp) # get the filtered amplitude back
            phi_cur = np.angle(F_tmp) # get the filtered phase back
            
            # derive the quad measurements from the filtered reconstruction
            I0_filtered = A_cur*np.cos(phi_cur)
            I270_filtered = A_cur*np.cos(phi_cur+pi/2)
            I180_filtered = A_cur*np.cos(phi_cur+pi)
            I90_filtered = A_cur*np.cos(phi_cur+pi*3/2)
            filtered_images = [I0_filtered, I90_filtered, I180_filtered, I270_filtered]
                        
        elif use_Gaussian == '1d' and sigma != 0:
            self.Gaussian = True
            self.sigma = sigma
            filter_axis = 1 if phase_variation_axis == 'x' else 0
            A = super().compute_amplitude_conventional(images)
            phi = super().compute_phase_conventional(images)
            
            F_tmp = gaussian_filter1d(A*(np.cos(phi) + 1j*np.sin(phi)), sigma, mode='constant', axis=filter_axis) # gaussian filter the hologram
            A_cur = 2*abs(F_tmp) # get the filtered amplitude back
            phi_cur = np.angle(F_tmp) # get the filtered phase back
            
            # derive the quad measurements from the filtered reconstruction
            I0_filtered = A_cur*np.cos(phi_cur)
            I270_filtered = A_cur*np.cos(phi_cur+pi/2)
            I180_filtered = A_cur*np.cos(phi_cur+pi)
            I90_filtered = A_cur*np.cos(phi_cur+pi*3/2)
            filtered_images = [I0_filtered, I90_filtered, I180_filtered, I270_filtered]
            
        self.composite_image = np.zeros(filtered_images[0].shape)
        
        if phase_variation_axis == 'x':
            for i in range(self.composite_image.shape[1]):
                self.composite_image[:,i] = filtered_images[i%num_phase_shifts][:,i]
        elif phase_variation_axis == 'y':
            for i in range(self.composite_image.shape[0]):
                self.composite_image[i,:] = filtered_images[i%num_phase_shifts][i,:]
        else:
            raise Exception("Only x or y phase variation axes are currently supported")
    
    """
        Mask and center the composite fft
        
        phase_variation_axis: 'x'|'y'
        
    """
    def calculate_fft_shifted(self, phase_variation_axis='x'):
        
        self.comp_img_32 = np.float32(self.composite_image) # make sure image is float 32
                        
        self.fft = np.fft.fftshift(np.fft.fft2(self.comp_img_32))  # Fourier transform of the composite image
        self.height, self.width = self.comp_img_32.shape
        
        self.fft_centered = np.zeros(self.fft.shape,  dtype=np.complex128)
        
        if self.k == 0:
            self.fft_centered = self.fft
        else:
            # Center the Fourier 
            if phase_variation_axis == 'y':
                shift_by = (self.k / pi) * (self.height / 2.0)
                self.fft_centered = shift(self.fft, (self.height/2.0, 0))
                self.fft_centered = shift(self.fft_centered, (-(self.height/2.0) + shift_by,0))

            elif phase_variation_axis == 'x': 
                shift_by = (self.k / pi) * (self.width / 2.0)
                self.fft_centered = shift(self.fft, (0, self.width/2.0))
                self.fft_centered = shift(self.fft_centered, (0, -(self.width/2.0) + shift_by))
                
            else:
                raise Exception("Only x or y phase variation axes are currently supported")
            
        self.fft_reverse = np.fft.ifft2(np.fft.ifftshift(self.fft_centered))

# Return the reconstructed intensity of the composite image
    def reconstruct_intensity(self):
        return np.abs(self.fft_reverse) / self.illum_intensity_scale * 2

# Return the reconstructed phase of the composite image
    def reconstruct_phase(self):
        return -np.angle(self.fft_reverse)
    
# Return the reconstructed intensity and phase using the n-bucket reconstruction method
    def reconstruct_n_bucket(self, phase_variation_axis='x'):
        bucket_count = int(360/np.rad2deg(self.k))
        height, width = self.composite_image.shape
        recon_intensity = np.zeros((height, width))
        recon_phase = np.zeros((height, width))
        
        if phase_variation_axis == 'y':
            # wrap the image to accommodate the last rows/columns
            captured_img_pad = np.zeros((height+bucket_count-1, width))
            captured_img_pad[:height, :] = self.composite_image
            captured_img_pad[-(bucket_count-1):, :] = self.composite_image[:bucket_count-1, :]
            self.n_bucket_image = captured_img_pad
            theta_n = np.linspace(0, 2*pi, num=bucket_count, endpoint=False).reshape((bucket_count, 1))
    
            for i in range(height):
                k_buckets = captured_img_pad[i:i+bucket_count, :].T
                cur_thetas = np.roll(theta_n, -(i%bucket_count))
        
                recon_intensity[i, :] = ((np.sqrt((k_buckets @ np.sin(cur_thetas))**2 + (k_buckets @ np.cos(cur_thetas))**2)) * 4/bucket_count).flatten()
                recon_phase[i, :] = (np.arctan2(k_buckets @ -np.sin(cur_thetas), k_buckets @ np.cos(cur_thetas)) - pi/2).flatten()
                
        else:
            # wrap the image to accommodate the last rows/columns
            captured_img_pad = np.zeros((height, width+bucket_count-1))
            captured_img_pad[:, :width] = self.composite_image
            captured_img_pad[:, -(bucket_count-1):] = self.composite_image[:, :bucket_count-1]
            self.n_bucket_image = captured_img_pad
            theta_n = np.linspace(0, 2*pi, num=bucket_count, endpoint=False).reshape((bucket_count, 1))
    
            for i in range(width):
                k_buckets = captured_img_pad[:, i:i+bucket_count]
                cur_thetas = np.roll(theta_n, -(i%bucket_count))
        
                recon_intensity[:, i] = ((np.sqrt((k_buckets @ np.sin(cur_thetas))**2 + (k_buckets @ np.cos(cur_thetas))**2)) * 4/bucket_count).flatten()
                recon_phase[:, i] = (np.arctan2(k_buckets @ -np.sin(cur_thetas), k_buckets @ np.cos(cur_thetas)) - pi/2.0).flatten()
        
        return recon_intensity, recon_phase
        
    
    def get_metadata(self):
        super().get_metadata()
        print("k = ", self.k, "rad ", np.degrees(self.k), "deg")
         