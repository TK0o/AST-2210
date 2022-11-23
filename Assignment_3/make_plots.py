import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from numba import vectorize, njit, prange
from numpy import nan
from matplotlib.patches import Rectangle

fit_data = fits.open('ADP.2017-03-27T12_08_50.541.fits')
# fit_data = fits.open('average.fits')
fit_data.info()
# arr = fit_data[0].data

def redshift(arr, z=0.005476):
    expected = arr * (1 + z)
    return expected

def print_diff(arr_res, arr_obs):
    for i in range (len(arr_res)):
        print (f"res {arr_res[i]:.4f} [Å], "
               f"exp obs {arr_obs[i]:.4f} [Å]")

def timeframe():
    from datetime import datetime
    obs_start = 56942.18866808
    obs_end = 56942.263365517
    tot_time = obs_end - obs_start
    sec_day = 3600 * 24
    
    time_list = []
    for i in range (1, 9):
        time_str = fit_data[0].header[f'PROV{i}']
        time_str = time_str.replace('MUSE.2014-10-12T', '')
        time_str = time_str.replace('.fits', '')
        time_str = time_str.replace('.', ':')
        time_list.append(datetime.strptime(time_str, "%H:%M:%S:%f"))
        
    time_diff = []
    for i in range (1, 8):
        time_diff.append((time_list[i] - time_list[i - 1]).total_seconds())
    
    print (f"1 [d] = {sec_day} [s]")
    print (f"tot observation time = {tot_time:.4f} [d]")
    print (f"0.0747 [d] = {sec_day * tot_time:.4f} [s]")
    print ("tot time between the first and last file "
           f"was created {sum(time_diff):.4f}")

def write_to_file(file):
    with open('h1.txt', 'w') as H1:
        H1.write(repr(file[1].header))
        
    with open('h2.txt', 'w') as H2:
        H2.write(repr(file[2].header))
        

e_line_name = []
a_line_name = []

with open('spectral_lines.txt', 'r') as s_lines:
    a = 0
    emission = []
    absorption = []
    s_lines.readline()
    for i in s_lines:        
        if i == 'absorption\n': 
            a = 1
        if a == 0 and i != '\n':
            emission.append(float(i.split()[0]))
            e_line_name.append(i.split()[-1])
        elif a >= 1 and i != 'absorption\n':
            absorption.append(float(i.split()[0]))
            a_line_name.append(i.split()[-1].replace('_', ' '))
    
    em = np.array(emission)
    ab = np.array(absorption)

def mean_flux(arr, idx_0=0, idx_1=0):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if idx_0 == 0 and idx_1 == 0:
            return np.nanmean(arr, 0)
        else:
            return np.nanmean(arr[idx_0:idx_1], 0)

@njit(parallel=True)
def fill_empty(arr):
    n_y = len (arr)
    n_x = len (arr[0])

    arr[245, 316] = 0
    for i in prange(n_y):
        for j in prange (n_x):
            if np.isnan(arr[i, j]):
                arr[i, j] = 0.1
    # return arr
    
def area(d, px, py):
    from math import ceil
    R = int(ceil((d - 1) / 2))
    area_arr = r_data[:, (py - R):(py + R + 1), (px - R):(px + R + 1)]
    return np.sum(np.sum(area_arr, 1), 1) / d**2

r_data = fit_data[1].data
center = np.loadtxt('center.txt')

def plot_flux_image1():
    arr = mean_flux(r_data, 1400, 1500)
    lvl = np.arange(1, 4.8, 0.005)
    fig, ax = plt.subplots(figsize=(8, 4))    

    im = ax.contour(np.log10(arr), origin='lower', levels=lvl, cmap='gray')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("log10(" + fit_data[0].header['BUNIT'] + ")")
    
    plt.title('NGC1365', fontsize=10, y=1.06)
    fig.text(0.15, 0.90, 'log10 of mean flux between $\lambda_0$ = ' +
                          f'{center[1400, 0]:.2f} [Å] to $\lambda_1$ = ' + 
                          f'{center[1500, 0]:.2f} [Å]')
    plt.ylabel('pixel')
    plt.xlabel('pixel')
    
    plt.savefig('../data/flux_image1.pdf')
    plt.savefig('../data/flux_image1.png')
    plt.show()

def plot_flux_image2():
    arr = mean_flux(r_data, 1350, 1550)    
    lvl = np.arange(1.2, 4.8, 0.001)
    fig, ax = plt.subplots(figsize=(8, 4))    
    
    im = ax.contourf(np.log10(arr), origin='lower', levels=lvl, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("log10(" + fit_data[0].header['BUNIT'] + ")")
    
    plt.title('NGC1365 sub areas', fontsize=10, y=1.06)
    fig.text(0.15, 0.90, 'log10 of mean flux between $\lambda_0$ = ' +
                          f'{center[1350, 0]:.2f} [Å] to $\lambda_1$ = ' + 
                          f'{center[1550, 0]:.2f} [Å]')
    plt.ylabel('pixel')
    plt.xlabel('pixel')
    ax.add_patch(Rectangle((160, 165), 10, -15, fc='none', color ='black',
                           linewidth = 1, linestyle="--"))
    ax.add_patch(Rectangle((185, 125), 10, 13, fc='none', color ='black',
                           linewidth = 1, linestyle="--"))
    ax.add_patch(Rectangle((76, 233), 10, 15, fc='none', color ='black',
                           linewidth = 1, linestyle="--"))
    ax.add_patch(Rectangle((189, 138), 10, 15, fc='none', color ='black',
                           linewidth = 1, linestyle="--"))
    
    ax.annotate('Nucleus', (130, 168))
    ax.annotate('Area B', (155, 110))
    ax.annotate('Area C', (45, 215))
    ax.annotate('Area D', (200, 155))

    plt.savefig('../data/flux_image2.pdf')
    plt.savefig('../data/flux_image2.png')
    plt.show()                         

def plot_flux_image3():    
    arr = mean_flux(r_data)
    lvl = np.arange(1.2, 4.8, 0.01)
    fig, ax = plt.subplots(figsize=(8, 4))    
    
    im = ax.contourf(np.log10(arr), origin='lower', levels=lvl, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("log10(" + fit_data[0].header['BUNIT'] + ")")
    
    plt.title('NGC1365', fontsize=10, y=1.06) 
    fig.text(0.15, 0.90, 'log10 of mean flux between $\lambda_0$ = ' +
                         f'{center[0, 0]:.2f} [Å] to ' +
                         f'$\lambda_1$ = {center[-1, 0]:.2f} [Å]')
    plt.ylabel('pixel')
    plt.xlabel('pixel')
    
    plt.savefig('../data/flux_image3.pdf')
    plt.savefig('../data/flux_image3.png')
    plt.show()

def plot_flux_image4():
    arr = mean_flux(r_data, 1400, 1500)
    lvl = np.arange(1, 4.8, 0.005)
    fig, ax = plt.subplots(figsize=(8, 4))    

    im = ax.contour(np.log10(arr), origin='lower', levels=lvl)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("log10(" + fit_data[0].header['BUNIT'] + ")")
    
    plt.title('NGC1365', fontsize=10, y=1.06)
    fig.text(0.15, 0.90, 'log10 of mean flux between $\lambda_0$ = ' +
                          f'{center[1400, 0]:.2f} [Å] to $\lambda_1$ = ' + 
                          f'{center[1500, 0]:.2f} [Å]')
    plt.ylabel('pixel')
    plt.xlabel('pixel')
    
    plt.savefig('../data/flux_image4.pdf')
    plt.savefig('../data/flux_image4.png')
    plt.show()

def plot_flux_image5():
    arr = mean_flux(r_data, 220, 230)
    lvl = np.arange(1, 4.8, 0.005)
    fig, ax = plt.subplots(figsize=(8, 4))    

    im = ax.contour(np.log10(arr), origin='lower', levels=lvl, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("log10(" + fit_data[0].header['BUNIT'] + ")")
    
    plt.title('NGC1365 [OIII]', fontsize=10, y=1.06)
    fig.text(0.15, 0.90, 'log10 of mean flux between $\lambda_0$ = ' +
                          f'{center[220, 0]:.2f} [Å] to $\lambda_1$ = ' + 
                          f'{center[230, 0]:.2f} [Å]')
    plt.ylabel('pixel')
    plt.xlabel('pixel')
    
    plt.savefig('../data/flux_image5.pdf')
    plt.savefig('../data/flux_image5.png')
    plt.show()
    
    
em_expected = redshift(em)
ab_expected = redshift(ab)

para = [[[3, 165, 155], [3e3, 5.5e3, 3e3, 1e4]], 
        [[3, 190, 130], [3e3, 3.75e3, 0.9e3, 2e3]],
        [[1, 81, 238], [1.25e3, 1.35e3, 0.8e3, 0.75e3]], 
        [[3, 194, 143], [3e3, 3.4e3, 1.5e3, 0.75e3]]]
       

def search_for_closest_idx(lines, empty_arr):
    n = len(center)
    for i, j in enumerate (lines):
        for k in range (len(center)):            
            if center[n - 1 - k, 0] - j <= 0:
                empty_arr[i] = n - 2 - k
                break
   
def find_peak(arr_data, j):
    ab_idx = np.zeros(7)
    em_idx = np.zeros(13)
    
    search_for_closest_idx(em_expected, em_idx) 
    search_for_closest_idx(ab_expected, ab_idx)
    
    ab_min = np.zeros(7)
    em_peak = np.zeros(13)
    
    for i in range (13):
        if i < 7:
            ab_min[i] = np.argmin(arr_data[int(ab_idx[i]-1):int(ab_idx[i] +1)])
        em_peak[i] = np.argmax(arr_data[int(em_idx[i] - 2):int(em_idx[i] + 2)])
      
    em_idx += em_peak - 2
    ab_idx += ab_min - 2
    
    if j == 0:
        em_idx[6] = 1468
        em_idx[8] = 1495
        ab_idx[4] = 439
        ab_idx[3] = 364
    return em_idx, ab_idx

def region_common(I, d, py, px, 
                  dot_adj_10, dot_adj_11, dot_adj_20, dot_adj_21):
    name = ['Nucleus, area A', 'area B', 'area C', 'area D']
    filename = ['area_A', 'area_B', 'area_C', 'area_D']
    arr = area(d, py, px)
    null_val = np.zeros(13)
    
    
    em_idx, ab_idx = find_peak(arr, I)

    fig, ax = plt.subplots(3, 2, figsize=(16, 12))
    
    ax[0, 0].plot(center[:, 0], arr[:], c='b', lw=0.75)
    
    if I == 0:
        ax[0, 1].plot(center[:1420, 0], arr[:1420], 'b', lw=0.75)
        ax[0, 1].plot(center[1625:, 0], arr[1625:], 'b', lw=0.75)
        ax[0, 1].plot(center[:0, 0], arr[:0], 'b', lw=0.75)
    elif I == 1:
        ax[0, 1].plot(center[130:1450, 0], arr[130:1450], 'b', lw=0.75)
        ax[0, 1].plot(center[1625:, 0], arr[1625:], 'b', lw=0.75)    
        ax[0, 1].plot(center[:110, 0], arr[:110], 'b', lw=0.75)
    else:
        ax[0, 1].plot(center[130:1450, 0], arr[130:1450], 'b', lw=0.75)
        ax[0, 1].plot(center[1625:3489, 0], arr[1625:3489], 'b', lw=0.75)    
        ax[0, 1].plot(center[:100, 0], arr[:100], 'b', lw=0.75)
        ax[0, 1].plot(center[3495:, 0], arr[3495:], 'b', lw=0.75) 
    
    ax[1, 0].plot(center[42:282, 0], arr[42:282], c='b', lw=0.75)
    
    ax[1, 1].plot(center[350:450, 0], arr[350:450], c='b', lw=0.75)
    
    ax[2, 0].plot(center[908:1306, 0], arr[908:1306], c='b', lw=0.75)
    
    ax[2, 1].plot(center[1450:1625, 0], arr[1450:1625], c='b', lw=0.75)

    lim = [[25e3, 12e3, 10e4], 
           [12e3, 5e3, 45e4], 
           [7e3, 3e3, 5e4], 
           [13e3, 3.75e3, 5e4]]
    
    d_adj = [dot_adj_10, dot_adj_11, dot_adj_20]
    q = 0
    for i, j in enumerate([[0, 3], [3, 6], [6, 11]]):
        for _, k in enumerate(em_idx[j[0]:j[1]]):
            k = int(k)
            X = center[k, 0]
            y_max = lim[I][i]
            ax[1 + q, i - q].vlines(x=X, ymin=arr[k] + 2.5e2, 
                                    ymax=y_max, color='k', ls='--', lw=0.75)
            ax[1 + q, i - q].annotate(e_line_name[_ + j[0]], 
                                       (X, y_max), (X + 2, y_max * 1.07))                              
        q = 1
    
    for i, j in enumerate([[1, 3], [3, 5], [5, 6]]):
        for _, k in enumerate(ab_idx[j[0]:j[1]]):
            k = int(k)
            X = center[k, 0]
            y_min = d_adj[i]
            ax[1 + (i > 1), i - 2 * (i > 1)].vlines(x=X, ymin=y_min, 
                                    ymax=arr[k], color='g', ls='--', lw=0.75)

            ax[1 + (i > 1), i - 2 * (i > 1)].annotate(a_line_name[_ + j[0]], 
                                        (X, y_min), (X + 4, y_min))       
    for i in range (3):
        for j in range (2):
            ax[i, j].set_xlabel('Wavelength [Å]')
            ax[i, j].set_ylabel(fit_data[0].header['BUNIT'])
    
    fig.suptitle(f'{name[I]}' , fontsize=20, y=0.925)
    fig.text(0.345, 0.89, f'{d}x{d} square centered at pixel $p_x$ = {px}' +
                 f' and $p_y$ = {py}', fontsize=15)
    plt.subplots_adjust(hspace=0.35)
    
    plt.savefig(f'../data/{filename[I]}.pdf')
    plt.savefig(f'../data/{filename[I]}.png')
    plt.show()    
    
    print()
    print ('\\begin{table}[H]')
    print ('\t\\centering')
    print ('\t\\caption{' + f'{name[I]}' + ' emission}')
    print ('\t\\begin{tabular}{c@{\\hspace{1cm}} c c c}')
    
    print('\t\t\\hline')
    print ('\t\tname & $\lambda$ [Å] at rest & '
           'expected $\lambda$ [Å] & measured $\lambda$ [Å] \\\\')
    print('\t\t\\hline')
    for i in range (11):
        print ('\t\t\\text{'+f'{e_line_name[i]}'+'} & ' + 
               f'{em[i]:.4f} & {em_expected[i]:.4f} &' + 
               f' {center[int(em_idx[i]), 0]:.4f} \\\\')
    print ('\t\t\\hline')
    print ('\t\\end{tabular}\\label{tab:midpointruletab}')
    print ('\end{table}')
    
    print()
    print ('\\begin{table}[H]')
    print ('\t\\centering')
    print ('\t\\caption{' + f'{name[I]}' + ' absorption}')
    print ('\t\\begin{tabular}{c@{\\hspace{1cm}} c c c}')
    print('\t\t\\hline')
    print ('\t\tname & $\lambda$ [Å] at rest & '
           'expected $\lambda$ [Å] & measured $\lambda$ [Å] \\\\')
    print('\t\t\\hline')
    for j in range (1, 6):
        print ('\t\t\\text{'+f'{a_line_name[j]}'+'} &' + 
               f' {ab[j]:.4f} & {ab_expected[j]:.4f} &' + 
               f' {center[int(ab_idx[j]), 0]:.4f} \\\\')
    print ('\t\t\\hline')
    print ('\t\\end{tabular}\\label{tab:midpointruletab}')
    print ('\end{table}')
    
# region_common(0, *para[0][0], *para[0][1])
def plot_regions():
    for i in range (4):
        region_common(i, *para[i][0], *para[i][1])

# timeframe()

# plot_regions()

# plot_flux_image1()
# plot_flux_image2()
# plot_flux_image3()
# plot_flux_image4()
plot_flux_image5()

fit_data.close()

def compare():
    h1 = open('h1.txt', 'r')
    h2 = open('h2.txt', 'r')
        
    l1 = [i[:-1] for i in h1]  
    l2 = [i[:-1] for i in h2]
    l = []
    
    for i in range (len(l1)):
        idx = [i]
        for j in range (len(l2)): 
            if l2[i] == l1[j]:
                idx.append(j) 
        l.append(idx)
    print (l)       
    
    h1.close()
    h2.close()