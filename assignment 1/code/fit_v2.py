from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

arr1 = np.load('idata_square.npy')
arr2 = np.load('spect_pos.npy')

def G(x, a, b, c, d):
    return a * np.exp(- (x - b)**2 / (2 * c**2)) + d

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
    
def adjust_coords(raw_data_array_yx, python_indexed_coords=False):
    size = np.shape(raw_data_array_yx)
    if len(size) == 1:
        raw_data_array_yx = raw_data_array_yx.reshape((1, 2))
        size = np.shape(raw_data_array_yx)
    
    ones = np.zeros(size, int)
    if python_indexed_coords == False:
        ones[:, :] = -1
    
    adjusted_data_array_yx = raw_data_array_yx + ones
    return adjusted_data_array_yx
    

def get_data(data_array_yx, x):
    size = np.shape(data_array_yx)
    data_set = np.zeros((size + np.array([0, 2])))
    j = 0
    for i in data_array_yx:
        arr = arr1[i[0], i[1]]
        data_set[j] = gauss_fit_fwhm(arr, np.float64(x))
        j += 1
    return data_set

def pre_curve_fit_fwhm(arr_in, x): 
    shape = np.shape(arr_in) - np.array([0, 0, 4])
    arr_out = np.zeros(shape, float)
    
    for i in range (len(arr_in)):
        for j in range (len(arr_in[0])):
            arr_out[i, j] = gauss_fit_fwhm(arr_in[i, j], x)
    return arr_out

def plot_data(coords, data_set, x, Color, Name, Label=0, include_average=0):        
    fig, ax = plt.subplots(figsize=(12, 6))
    X = np.linspace(x[0], x[-1], 100)
    plt.title('Spectra')
    j = 0
    for i in data_set:
        arr = arr1[coords[j, 0], coords[j, 1]]
        popt, pcov = curve_fit(G, x, arr, i, maxfev = 10000)
                
        ax.plot(x, arr, marker='.', ls='--', color=Color[j], 
                label=Name[j], lw=1, markersize=5)         
        
        ax.plot(X, G(X, *popt), color=Color[j], 
                label='Gauss fit ' + Name[j], lw=1)
        j += 1

    if include_average != 0:
        average = sum(sum(arr1)) / (len(arr1) * len(arr1[0]))

        popt, pcov = curve_fit(
            G, x, average, gauss_fit_fwhm(average, x), maxfev = 10000)
            
        ax.plot(x, average, marker='.', ls='--', color='purple', 
                label='average', lw=1, markersize=5)      
        ax.plot(X, G(X, *popt), color='purple', label='Gauss fit ' + 'average')

    if Label != 0 and include_average != 0:
        ax.legend(bbox_to_anchor=(0.5, -0.25), loc='lower center', ncol=5)
        
    elif Label != 0 and include_average == 0:
        ax.legend(bbox_to_anchor=(0.5, -0.25), loc='lower center', ncol=4)
    plt.ylabel('intensity [$Wm^{-2}$]')
    plt.xlabel('$\lambda$ in [Å]')
    # plt.xlabel('$\lambda$ in [Å]')
    plt.savefig('Spectra.pdf')
    plt.show()

def plot_data_simple(coords, data_set, x, include_average=0):
    fig, ax = plt.subplots(figsize=(12, 6))
    X = np.linspace(x[0], x[-1], 100)
    arr_out = np.zeros((len(coords), 4))
    j = 0
    for i in data_set:
        arr = arr1[coords[j, 0], coords[j, 1]]
        popt, pcov = curve_fit(G, x, arr, i, maxfev = 10000)
        arr_out[j] = popt
        ax.plot(x, arr, marker='.', ls='--', lw=1, markersize=5)         
        
        ax.plot(X, G(X, *popt), lw=1)
        j += 1

    if include_average != 0:
        average = sum(sum(arr1)) / (len(arr1) * len(arr1[0]))
        
        popt, pcov = curve_fit(
            G, x, average, gauss_fit_fwhm(average, x), maxfev = 10000)
            
        ax.plot(x, average, marker='.', ls='--', color='purple', 
                label='average', lw=1, markersize=5)      
        ax.plot(X, G(X, *popt), color='purple', label='Gauss fit ' + 'average')
    plt.xlabel('$\lambda_i$ in [Å]')
    plt.ylabel('intensity I')
    plt.show()
    return arr_out

def get_values(arr, x):
    average = sum(sum(arr1)) / (len(arr1) * len(arr1[0]))
    popt, pcov = curve_fit(
            G, x, average, gauss_fit_fwhm(average, x), maxfev = 10000)
    
    b = np.zeros((len(arr) + 1))
    b[-1] = popt[1]
    pre_arr = get_data(arr, x)
    j = 0
    # print (pre_arr)
    for i in pre_arr:
        arr_ = arr1[arr[j, 0], arr[j, 1]]
        popt, pcov = curve_fit(G, x, arr_, i, maxfev = 10000)
        b[j] = popt[1]
        j += 1
    return b    

def get_doppler(arr):
    lambda0 = 6173
    vel = ((arr - lambda0) / lambda0) * 3e8
    return vel


color = ['r', 'b', 'g', 'orange']
name = ['A', 'B', 'C', 'D']

coords = np.array([[197, 49],
                   [443, 238],
                   [213, 397],
                   [52, 466]])

coords = adjust_coords(coords)

data_set = get_data(coords, arr2)
# plot_data(coords, data_set, arr2, color, name, 1, 1)
val = get_values(coords, arr2)
vel = get_doppler(val)
# print (val)
# print (vel)
for i in range (len(vel)):
    print (f"{vel[i]:.5f} & {val[i]:.5f} \\\\")