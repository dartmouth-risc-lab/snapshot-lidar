import numpy as np
from scipy.constants import c, pi
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter, gaussian_filter1d, shift

'''
    Helper class for calculating integration
'''


""" Convert phase variation increment (in degrees) to number of rows/columns cropped
"""
def degrees2crop(degrees_per_line, size=640):
    overflow = size % (360 / degrees_per_line)
    return int((360 / degrees_per_line) + overflow) if overflow != 0 else 0 

# MLX two's complement function as found in the Python SDK
def mlx_two_comp(phase_in):
    phase_tmp1 = phase_in.astype(np.uint16)
    phase_tmp2 = np.left_shift(phase_tmp1, 4)
    phase_tmp3 = phase_tmp2.astype(np.int16)
    phase_tmp4 = np.right_shift(phase_tmp3, 4)
    phase_out = phase_tmp4.astype(np.double)
    return phase_out

def compute_I_analytical(A, theta, phi, method=''):
    if method=='mlx':
        return (A/2.0) * np.cos((pi/2.0) - theta - phi)
    else:
        return (A/2.0) * np.cos(theta - phi)

'''
    Scale array to (0, 1).
  
'''
def scale_min_max(array):
    if (np.max(array) - np.min(array)) == 0:
        return array
    return (array - np.min(array)) / (np.max(array) - np.min(array))

'''
    Scale array to (minmax[0], minmax[1]).
  
'''
def scale_to_range(array, minmax):
    return scale_min_max(array) * (minmax[1] - minmax[0]) + minmax[0]

'''
    Calculate the root mean squared error between array1 and array2.
  
'''
def rmse(array1, array2):
    return np.sqrt((array1 - array2)**2 / (array1.shape[0] * array1.shape[1]))

'''
    Return a circular masked img for rotation analysis.

    img: input image
    center: (x, y) coordinates of the center of the circle
    radius: radius of the circle, in pixels
  
'''
def create_mask(img, center, radius):
    h, w = img.shape

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[1])**2 + (y-center[0])**2)

    mask = dist_from_center <= radius
    return mask * img

'''
    Create a gaussian mask in the fourier domain for the composite fft.

    h, w: height and width of the image
    k: radians per row or column
  
'''
def create_gaussian_mask(h, w, k, sigma=10):
    if sigma==0:
        return np.ones((h, w))
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    m = filter_size//2
    n = filter_size//2

    shift = int(k * w / (2*pi))
    gaussian_disk = np.zeros((480, 640))
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            if h//2+x<0 or h//2+x>=h or w//2+y<0 or w//2+y>=w:
                continue
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_disk[240+x, 320+y] = (1/x1)*x2
    gaussian_disk /= np.max(gaussian_disk)

    mask = np.zeros(gaussian_disk.shape)
    mask[:, :w//2] = gaussian_disk[:, shift:shift+w//2]
    mask[:, w//2:] = gaussian_disk[:, w//2-shift:w-shift]

    return mask

'''
    Calculate the snr, ignoring the last n_bucket rows and columns from n bucket wrapping

    expected: ground truth image
    measured: measured image
    n_bucket: the x rows and y columns to exclude from snr calculation
  
'''
def calculate_snr(expected, measured, n_bucket=(0, 0)):
    if n_bucket[0] != 0:
        expected_mask = expected[:-(n_bucket[0]-1), :]
        measured_mask = measured[:-(n_bucket[0]-1), :]
    elif n_bucket[1] != 0:
        expected_mask = expected[:, :-(n_bucket[1]-1)]
        measured_mask = measured[:, :-(n_bucket[1]-1)]
    else:
        expected_mask = expected
        measured_mask = measured

    return 10*np.log10(np.mean(expected_mask**2)/np.mean((expected_mask - measured_mask)**2))