from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
import pickle

def G(x, a, b, c, d):
    return a * np.exp(- (x - b)**2 / (2 * c**2)) + d

@njit
def gauss_fit_fwhm(data, x):
    n = len (data)
    d = np.max(data)
    a = np.min(data) - d
    
    b_i = np.argmin(data)
    b_im = b_i - (b_i > 0)
    b_ip = b_i + 1 + (b_i < n - 1)
    b = np.mean(x[b_im:b_ip])

    y = d + a / 2
    x_r1, y_r1 = 0, 0
    x_r2, y_r2 = 0, d
    for i in range (b_i, n):
        if y_r1 < data[i] < y:
            y_r1 = data[i]
            x_r1 = i
        if y_r2 > data[i] > y:
            y_r2 = data[i]
            x_r2 = i 
            
    if data[0] > y and b_i > 1:
        x_l1, y_l1 = 0, data[0]
        x_l2, y_l2 = 0, 0
        for i in range (b_i):
            if y_l1 > data[i] > y:
                y_l1 = data[i]
                x_l1 = i
            if y_l2 < data[i] < y:
                y_l2 = data[i]
                x_l2 = i
    else:
        x_l1, y_l1 = 0, data[0]
        x_l2, y_l2 = 1, data[1]
    
    intersection1 = (y - y_r1) / ((y_r2 - y_r1)/(x[x_r2] - x[x_r1])) + x[x_r1]    
    intersection2 = (y - y_l1) / ((y_l2 - y_l1)/(x[x_l2] - x[x_l1])) + x[x_l1]
            
    FWHL = abs(intersection1 - intersection2)
    c =  FWHL / np.sqrt(8 * np.log(2))

    return a, b, c, d

@njit(parallel=True)
def pre_curve_fit(arr_in, x):
    arr_out = np.zeros((len(arr_in), len(arr_in[0]), 4))
    
    for i in prange (len(arr_in)):
        for j in prange (len(arr_in[0])):
            arr_out[i, j] = gauss_fit_fwhm(arr_in[i, j], x)
    print ('pre_curve_fit complete')
    return arr_out

def curve_fit_region(arr_y, arr_guess, x):
    curve_fit_val = np.zeros(np.shape(arr_guess), float)
    for i in prange (len(arr_y)):
        for j in prange (len(arr_y[0])):
            popt, pcov = curve_fit(G, x, arr_y[i, j], 
                                   arr_guess[i, j], maxfev = 1000)
            curve_fit_val[i, j] = popt
    return curve_fit_val

def generate_data():
    arr_guess = pre_curve_fit(arr1, arr2)
    arr = curve_fit_region(arr1, arr_guess, arr2)
    pickle.dump(arr, open('lambda_val_fwhl_v2.dat', 'wb'))
    return arr

def load_data():
    return pickle.load(open('lambda_val_fwhl_v2.dat', 'rb'))

def get_doppler(arr):
    lambda0 = 6173
    vel = ((arr - lambda0) / lambda0) * 3e8
    return vel

def average_c_val():
    arr = load_data()
    c_val = arr[:, :, 2]
    C_val = c_val.reshape(np.size(c_val))
    print (f'average value of c = {np.mean(C_val)}')


def add_dots(dsize = 5):
    name = ['A', 'B', 'C', 'D']
    coords = np.array([[197, 49],
                       [443, 238],
                       [213, 397],
                       [52, 466]])
    
    for l, k in enumerate(coords):
        plt.plot(k[1], k[0], '.', color='r', markersize=dsize)
        plt.annotate(name[l], (k[1], k[0]), (k[1] + 10, k[0] + 1))

def doppler_sub_field_view(x1=525, y1=325, h=100, w=150, 
                           python_indexed_coords=False):
    if python_indexed_coords == False:
        x1 = x1 - 1
        y1 = y1 - 1
    x2 = x1 + w
    y2 = y1 + h
    arr_sub_field_of_view = load_data()[y1:y2, x1:x2, :]
    vel = get_doppler(arr_sub_field_of_view[:, :, 1])
    
    fig, ax = plt.subplots(figsize=(8, 4)) 
    im = ax.imshow(vel, origin='lower', extent=[x1, x2, y1, y2])
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r"velocity $[m/s]$")
    
    ax.set_title('doppler velocity, sub field of view')
    ax.set_xlabel("x [idx]")
    ax.set_ylabel("y [idx]")
    
    
    
    
    plt.savefig('full_doppler.pdf')
    fig.tight_layout()
    plt.show()
    
def full_doppler():
    arr = load_data()
    vel = get_doppler(arr[:, :, 1])

    fig, ax = plt.subplots(figsize=(8, 4))    
    im = ax.imshow(vel, origin='lower')
    add_dots()
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(r"velocity $[m/s]$")
    ax.set_title('doppler velocity full field of view')
    ax.set_xlabel("x [idx]")
    ax.set_ylabel("y [idx]")
    plt.savefig('full_doppler.pdf')
    fig.tight_layout()
    plt.show()

arr1 = np.float64(np.load('idata_square.npy'))
arr2 = np.float64(np.load('spect_pos.npy'))


# generate_data()
full_doppler()
# doppler_sub_field_view()



