from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
import pickle

arr1 = np.load('idata_square.npy')
arr2 = np.load('spect_pos.npy')

def get_doppler(arr):
    lambda0 = 6173
    vel = ((arr - lambda0) / lambda0) * 3e8
    return vel

def load_data():
    return pickle.load(open('lambda_val_fwhl_v2.dat', 'rb'))

lamda = load_data()[:, :, 1]
vel = get_doppler(lamda)

average = np.sum(arr1, 2)/8
# np.fla

# plt.plot(average, vel, '.', color='b', markersize=0.1)
# plt.hist(average.flatten())
plt.plot(vel, average, '.', color='b', markersize=0.1)
plt.xlabel('velocity [m/s]')
plt.ylabel('intensity [$Wm^{-2}$]')
plt.title('continuum intensity vs doppler velocity')
plt.savefig('continuum_intensity_vs_doppler_velocity.pdf')
plt.show()



# plt.hist(vel)
# plt.show()

# for i in range (8):
#     plt.plot(vel, arr1[:, :, i], '.', color='b', markersize=0.1)
#     plt.title(f'spec nr {i}, $\lambda$ = {arr2[i]}')
#     plt.show()