# Basic utilities and file operations
import os
import sys
import shutil
from scipy.signal import find_peaks
# Numerical and scientific computing
import numpy as np  # For numerical operations
import scipy  # For scientific computing and advanced mathematical operations
from scipy import signal  # For signal processing tasks
from scipy.optimize import curve_fit, least_squares, leastsq, fsolve  # For optimization and fitting
from scipy.interpolate import interp1d  # For interpolation
from scipy.ndimage import label  # For image labeling
import math  # For basic mathematical operations
import statistics  # For statistical operations

# Data handling and analysis
import pandas as pd  # For data manipulation and analysis
from collections import Counter  # For counting hashable objects
from scipy import stats  # For statistical functions
from scipy.signal import chirp, find_peaks, peak_widths
# Image processing and computer vision
import cv2  # For computer vision tasks
from PIL import Image  # For image manipulation
from skimage import io, color, measure  # For image I/O and processing
from skimage.transform import hough_circle, hough_circle_peaks  # For Hough circle transforms
from skimage.feature import canny  # For edge detection
from skimage.draw import circle_perimeter  # For drawing circle perimeters
from skimage.util import img_as_ubyte  # For image type conversion

# Plotting and visualization
import matplotlib.pyplot as plt  # For plotting and visualization

# Machine learning
from sklearn.linear_model import RANSACRegressor, LinearRegression  # For regression analysis
from sklearn.pipeline import make_pipeline  # For making pipelines
from sklearn.preprocessing import PolynomialFeatures  # For generating polynomial features
from sklearn.metrics import mean_squared_error  # For calculating mean squared error

# Astronomy specific packages
import astropy  # For astronomy specific tasks
from astropy.io import fits  # For handling FITS files
from astropy.time import Time  # For handling time objects
from astropy.coordinates import SkyCoord  # For celestial coordinate system conversions
import astropy.units as u  # For unit conversions

# Polynomial operations
from numpy.polynomial.polynomial import polyder  # For polynomial differentiation
from numpy.polynomial import Polynomial  # For polynomial operations






def rotate_calib(data,theta=None):
    shape = (data.shape[1],data.shape[0])
    center = (int(len(data[0])/2),int(len(data)/2))
    matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 ) #поворот матрицы вокруг указанного центра
    data_rotated = cv2.warpAffine(src=data, M=matrix, dsize=shape, borderValue=65535) #сохранение повернутого изображения из матриц  
    data_rotated = crop_after_rotating(data_rotated)  

    return data_rotated

def crop_after_rotating(data_rotated):

    #################################  
    #CROPPING AFTER ROTATION
    #################################    
    #left upper
    
    value = 65535
    
    index_vertical = 0
    
    for j in range(0,len(data_rotated)):
        if data_rotated[j][0] != value:
            index_vertical = j
            if index_vertical>len(data_rotated)*0.3:
                index_vertical = int(len(data_rotated)*0.25)
            break
    data_rotated = data_rotated[index_vertical:]
    #right bottom
    index_vertical = 0
    for j in range(1,len(data_rotated)):
        if data_rotated[len(data_rotated)-j][-1] != value:
            index_vertical = j
            if index_vertical>len(data_rotated)*0.3:
                index_vertical = int(len(data_rotated)*0.25)
            break
    data_rotated = data_rotated[:len(data_rotated)-index_vertical]
    
    #plt.figure(figsize=(10,8))
    #plt.imshow(data_rotated)
    #plt.title("right bottom")
    
    #left bottom
    index_horizontal = len(data_rotated[0])
    for i in range(len(data_rotated[0])):
        if data_rotated[len(data_rotated)-1][i] != value: ## ==
            index_horizontal = i
            if index_horizontal>len(data_rotated[0])*0.3:
                index_horizontal = int(len(data_rotated[0])*0.25)
            break
    data_rotated = data_rotated[:,index_horizontal:]
    #plt.figure(figsize=(10,8))
    #plt.imshow(data_rotated)
    #plt.title("left bottom")
    
    #right upper
    index_horizontal = 0
    for i in range(len(data_rotated[0])):
        if data_rotated[0][len(data_rotated[0])-1-i] != value:
            index_horizontal = i
            break
    data_rotated = data_rotated[:,:len(data_rotated[0])-1-index_horizontal]
    #---------------------------------------------------------------------
    #plt.figure(figsize=(10,8))
    #plt.imshow(data_rotated)
    #plt.title("right upper")
    
    #right upper
    index_vertical = 0
    for j in range(0,len(data_rotated)):
        if data_rotated[j][-1] != value:
            index_vertical = j
            if index_vertical>len(data_rotated)*0.3:
                index_vertical = int(len(data_rotated)*0.25)
            break
    data_rotated = data_rotated[index_vertical:]
    
    #left bottom
    index_vertical = 0
    for j in range(1,len(data_rotated)):
        if data_rotated[len(data_rotated)-j][0] != value:
            index_vertical = j
            if index_vertical>len(data_rotated)*0.3:
                index_vertical = int(len(data_rotated)*0.25)
            break
    data_rotated = data_rotated[:len(data_rotated)-index_vertical]
    #left upper
    index_horizontal = 0
    for i in range(len(data_rotated[0])):
        if data_rotated[0][i] != value:
            index_horizontal = i
            break
    data_rotated = data_rotated[:,index_horizontal:]
    #right bottom
    index_horizontal = 0
    for i in range(len(data_rotated[0])):
        if data_rotated[len(data_rotated)-1][len(data_rotated[0])-1-i] != value:
            index_horizontal = i
            if index_horizontal>len(data_rotated[0])*0.3:
                index_horizontal = int(len(data_rotated[0])*0.25)
            break
    data_rotated = data_rotated[:,:len(data_rotated[0])-1-index_horizontal]
    
    
    
    #plt.figure(figsize=(10,8))
    #plt.imshow(data_rotated, cmap='gray')
    #plt.title(f"Pre-Croped image")
    #plt.grid()
#
    return data_rotated

def rotate_spectra(data):
    # Вспомогательная функция для вычисления ширины пика
    
    def measure_peak_width(data):
        value = np.mean(data)/2
        mask = np.where(data < value, 1,0.)  # Предполагается, что переменная `value` задана где-то выше
        sum_row = mask.sum(axis=1)
        # Скользящее среднее
        N_rm = 60
        W = np.zeros_like(sum_row); W[0:N_rm] = 1
        FConv = np.fft.fft(sum_row) * np.fft.fft(W)
        TConv = np.fft.ifft(FConv).real / N_rm
        X_rm = np.pad(TConv[(N_rm-1)//2:], (0, (N_rm-1)//2), 'constant')
        peaks, _ = find_peaks(X_rm, distance=None)
        
        # closest_peak_index = find_peak_closest_to_center(X_rm, peaks)

        # Если вам нужна высота каждого пика, вы можете ее измерить, используя значения в X_rm
        peak_heights = X_rm[peaks]
        
        # Теперь, чтобы найти индекс максимального пика, сначала найдем индекс максимальной высоты
        index_of_max_peak = peak_heights.argmax()
        
        # Используем этот индекс, чтобы найти индекс максимального пика в исходном массиве
        index_in_X_rm = peaks[index_of_max_peak]

        
        result_full = peak_widths(X_rm, np.array([index_in_X_rm]), rel_height=0.9)
        return result_full, X_rm  # Возвращаем ширину пика и обработанные данные

    def find_peak_closest_to_center(X_rm, peaks):
        center = len(X_rm) // 2
        closest_peak_index = min(peaks, key=lambda x: abs(x-center))
        return closest_peak_index

    def search_theta(data,start=1,stop=90):
        best_width = np.inf
        best_theta = None
        data_rotated_best = None
        
        # Поиск идеального угла с шагом 1 градус
        for theta in np.arange(start, stop):

            data_rotated = rotate_calib(data, theta=theta)

            result_full, xrm = measure_peak_width(data_rotated)
            width = result_full[0][0]
            if width < best_width:
                best_width = width
                best_theta = theta
                data_rotated_best = data_rotated

            elif width > best_width:
                break  # Ширина начала увеличиваться, останавливаем поиск

        # Уточнение угла с шагом 0.1 градуса
        for theta in np.arange(best_theta-1, best_theta+1,0.1):

            data_rotated = rotate_calib(data, theta=theta)
            result_full, xrm = measure_peak_width(data_rotated)
            width = result_full[0][0]
            if width < best_width:
                best_width = width
                best_theta = theta
                data_rotated_best = data_rotated
        return best_width, best_theta, data_rotated_best
       
    best_width, best_theta, data_rotated_best = search_theta(data,start=1,stop=90)

    if best_theta == 0:
        best_width, best_theta, data_rotated_best = search_theta(data,start=-5,stop=0)
        
        
    fig = plt.figure(figsize=(8,6),dpi = 100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(data, cmap='gray')
    ax1.set_title(f"Initial image")
    ax1.grid(alpha=0.5)

    ax2 = fig.add_subplot(1,2,2)   
    ax2.imshow(data_rotated_best, cmap='gray')
    ax2.set_title(f"Opt. rot. image ({round(best_theta,1)} deg)")
    ax2.grid(alpha=0.5)
    
    return data_rotated_best

def poly_mse_new(x,y):
    
    MAX_DEGREE = 15
    candidates = [] 
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    for order_mse in range(1, MAX_DEGREE+1):
        mse, poly = get_poly_mse(x_sorted, y_sorted, order_mse)
        y_pred = poly(x_sorted)

        if is_strictly_monotonic(y_pred):
            smoothness = is_smooth(poly, x_sorted)
            candidates.append((order_mse, poly, smoothness))

    if candidates:
        sorted_candidates = sorted(candidates, key=lambda x: x[2])
        median_candidate = sorted_candidates[len(sorted_candidates) // 2]
        best_poly = median_candidate[1]
    else:
        best_poly = None


    if best_poly is not None:
        print(f"Best degree: {median_candidate[0]}, MSE: {mse}")
        plt.figure(figsize=(15, 10))
        plt.scatter(x_sorted, y_sorted, color='blue', s=20, marker='o', label='Actual Data')
        x_test = np.linspace(min(x_sorted), max(x_sorted), 500)
        plt.plot(x_test, best_poly(x_test), color='red', label=f'Best polynomial of degree {median_candidate [0]}')
        plt.legend(loc='upper left')
        plt.show()
    else:
        print("No suitable monotonous polynomial was found.") 
    
    def poly_function(x):
        return sum(c * x**i for i, c in enumerate(best_poly.coef[::-1]))

    # Используем fsolve для нахождения корня полинома, начиная с близкого к нулю значения (например, 1)
    root = fsolve(poly_function, 1)[0]
    
    #extrapolation
    x_points = [0, x_test[0]]
    y_points = [0, best_poly(x_test)[0]]
    
    extrapolation_poly = Polynomial.fit(x_points, y_points, 20).convert()
    extrapolation_coefficients = extrapolation_poly.coef
    
    def polynomial_extrapolate(x):
        return sum(c * x**i for i, c in enumerate(extrapolation_coefficients))
    
    x_extrapolate = np.linspace(0, x_test[0], 100)
    y_extrapolate = [polynomial_extrapolate(x_val) for x_val in x_extrapolate]
        
    x_full = np.concatenate([x_extrapolate,x_test])
    y_full = np.concatenate([y_extrapolate,best_poly(x_test)])

    plt.figure(figsize=(15, 10))
    plt.scatter(x_sorted, y_sorted, color='blue', label='actual data')
    plt.plot(x_full, y_full, 'r--', label='Polynomially Extrapolated Data')
    plt.legend()
    plt.show()
    
    return best_poly,polynomial_extrapolate,root



def calib_preob(data):
    array_log10 = np.log10(65535/data)*1000 
    mean = array_log10.mean(axis=0)
    return mean

def graf(data, order):
    fig = plt.figure(figsize=(8,6),dpi = 100)
    ax1 = fig.add_subplot(2,1,1)
    x = np.arange(1, len(data) + 1)
    ax1.scatter(x,data,color = 'orangered', s = 2, alpha = 0.5)
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Value of pixel")

    peak_indexes,peak_mean = peaks(data,order=order) #func_peaks
    ax2 = fig.add_subplot(2,1,2)
    x1 = np.arange(0,len(data))
    ax2.plot(x1,data)
    ax2.scatter(peak_indexes, data[[peak_indexes]], color = 'y', s = 10, marker = 'D', label = 'maxima')
    ax2.scatter(peak_indexes, peak_mean, color = 'r', s = 10, marker = 'D', label = 'maxima')
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Value of pixel")
    
    return peak_indexes,peak_mean

def peaks(mean,order):
    """
    function for searching peaks in array 
    """
    peak_indexes = signal.argrelextrema(mean, np.greater, order=order)[0] #индексы пиков 
    peak_mean =[] # создаем пустой массив для средних значений пиков
    for i in range(0,len(mean)):
        if i in peak_indexes:
            peak_mean.append((mean[i-2]+mean[i-1]+mean[i]+mean[i+1]+mean[i+2])/5) # ищем среднее по пяти рядом стоящих значений
    #возвращает позицию/индекс найденного пика и среднее значение по пяти значениям : пик + 2 слева + 2 справа
    return peak_indexes,peak_mean



def mean_deviation(x_array,y_array):

    c = []
    for i in range(1,len(y_array)):
        c.append(abs(y_array[i] - y_array[i-1]))
    mean_deviation = np.array(c).sum()/len(y_array)

    #now we sort our data: if delta between neighboring elements larger than 2 mean deveation we will remove it
    indexes = []
    for i in range(1,len(y_array)):
        if abs(y_array[i]-y_array[i-1])>mean_deviation*2:
            indexes.append(i)

    y_array = np.delete(y_array,indexes)
    x_array = np.delete(x_array,indexes)
    xarray_flux = 10000/10**(x_array/2.5)
    a = []
    for i in range(0,len(xarray_flux)):
        a.append([xarray_flux[i],y_array[i]])

    a.sort(key=lambda x:x[0])

    xarray_flux = []
    y_array = []

    for i in range(0,len(a)):
        xarray_flux.append(a[i][0])
        y_array.append(a[i][1])

    xarray_flux = np.array(xarray_flux)
    y_array = np.array(y_array)       
    del a
    
    plt.figure(figsize=(6,4),dpi=120)
    plt.plot(xarray_flux, y_array, markersize=3,marker='o' )
    plt.xlabel("Интенсивность")
    plt.ylabel("Почернение")
    
    return xarray_flux, y_array



def poly_mse(x,y):
    MAX_DEGREE = 15
    candidates = [] #####
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    for order_mse in range(1, MAX_DEGREE+1):
        mse, poly = get_poly_mse(x_sorted, y_sorted, order_mse)
        y_pred = poly(x_sorted)

        if is_strictly_monotonic(y_pred):
            smoothness = is_smooth(poly, x_sorted)
            candidates.append((order_mse, poly, smoothness))


    if candidates:
        sorted_candidates = sorted(candidates, key=lambda x: x[2])
        median_candidate = sorted_candidates[len(sorted_candidates) // 2]
        best_poly = median_candidate[1]
    else:
        best_poly = None

    if best_poly is not None:
        print(f"Best degree: {median_candidate[0]}, MSE: {mse}")

        plt.scatter(x_sorted, y_sorted, color='blue', s=20, marker='o', label='Actual Data')
        x_test = np.linspace(min(x_sorted), max(x_sorted), 500)
        plt.plot(x_test, best_poly(x_test), color='red', label=f'Best polynomial of degree {median_candidate [0]}')
        plt.xlabel('dar')
        plt.legend(loc='upper left')
        plt.show()
    else:
        print("No suitable monotonous polynomial was found.") 
        
    return best_poly,x_test, x_sorted, y_sorted


def is_smooth(poly, x):
    derivative = poly.deriv()
    derivative_values = derivative(x)
    smoothness = np.std(derivative_values)
    return smoothness

def get_poly_mse(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
#     print('coef',coeffs)
    poly = np.poly1d(coeffs)
#     print("poly",poly)
    y_pred = poly(x)
    mse = ((y - y_pred) ** 2).mean()
    return mse, poly

def is_strictly_monotonic(y_vals):
    diff = np.diff(y_vals)
    return np.all(diff > 0) or np.all(diff < 0)


#weight approx
def align_20240118_weight_mask(data_full, data_mask):
    plt.figure()
    plt.plot(data_mask.sum(axis=1))
    sum_row = data_mask.sum(axis=1)
    N_rm = len(sum_row) // 20
    W = np.zeros_like(sum_row); W[0:N_rm] = 1
    FConv = np.fft.fft(sum_row) * np.fft.fft(W)
    TConv = np.fft.ifft(FConv).real / N_rm
    X_rm = np.pad(TConv[(N_rm-1)//2:], (0, (N_rm-1)//2), 'constant')
    plt.plot(X_rm)
    plt.title("Peaks mask")
    peaks, _ = find_peaks(X_rm, distance = len(X_rm)/10)
    peaks_ = []
    mean_ = np.mean(X_rm)
    for i in range(len(peaks)):
        if X_rm[peaks[i]]>mean_:
            peaks_.append(peaks[i])
    peaks = np.array(peaks_).astype("int")        
    
    plt.scatter(peaks,X_rm[peaks],color="red")
    
    plt.axhline(mean_)

    if len(peaks) in [1,2] :
        results_full = peak_widths(X_rm, [peaks[-1]], rel_height=.96) #peak_widths
        plt.hlines(*results_full[1:], color="purple")
        aligned_image = polynomial_align_lamp(data_full,int(results_full[-2][0]),int(results_full[-1][0]))

    elif len(peaks) == 3:
        closest_to_center = find_peak_closest_to_center(X_rm,peaks)
        if closest_to_center:
            results_full = peak_widths(X_rm, [closest_to_center], rel_height=.96) #peak_widths
            plt.hlines(*results_full[1:], color="purple")
            
            aligned_image = polynomial_align_spectra(data_full,int(results_full[-2][0]),int(results_full[-1][0]))
            plt.figure()
            plt.imshow(aligned_image)
        else:
            raise ValueError("There is no peak near the center")

    return aligned_image#, int(results_full[-2][0]),int(results_full[-1][0])
    
    
    
    
def polynomial_align_lamp(data_full, start, stop):    
    clean_data = np.where(data_full < np.mean(data_full)*.5, 0.00001, 1) #last_change
    data_part = clean_data[start:stop]
    clean_data = data_part
    plt.figure()
    plt.imshow(clean_data)
    plt.title("CHECK DATA PART")
    x_coordinates, y_coordinates = get_indexis_from_lamp(data_part) 
    fig,ax1 = plt.subplots(figsize=(8,5),dpi=100)
    ax1.imshow(data_full, cmap='gray')   
    xp1 = np.arange(0,len(data_full[0]))
   
    fig,ax2 = plt.subplots(figsize=(8,5),dpi=100)
    ax2.imshow(data_part, cmap='gray')  
    ax2.set_title("data_part")

    #weights of values
    len_x = len(x_coordinates)
    previous_mse = None

    # Создаем модель RANSAC для аппроксимации полиномом с использованием пайплайна
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    ransac = RANSACRegressor(base_estimator=poly_model, 
                             min_samples=int(len_x * 0.8),  # Минимальное количество точек для принятия модели
                             residual_threshold=2,         # Порог для определения выбросов
                             random_state=42)              # Зафиксируем случайное состояние для воспроизводимости
    
    # Преобразуем данные для использования в модели RANSAC
    X = np.array(x_coordinates).reshape(-1, 1)  # Преобразуем x_coordinates в двумерный массив
    
    # Подгоняем модель к данным
    ransac.fit(X, y_coordinates)
    
    # Получаем параметры полинома
    z1 = np.polyfit(x_coordinates, y_coordinates, 3)
    
    polynomial1 = np.poly1d(z1)
    y_polynomial1 = polynomial1(xp1)
    
    
    xp_tg1 = np.arange(0,len(y_polynomial1))
    z_tg1 = np.polyfit(xp_tg1,y_polynomial1, 1)
    polynomial_tg1 = np.poly1d(z_tg1)
    y_polynomial_tg1 = polynomial_tg1(xp_tg1)
    tg1 = polynomial_tg1[1]
    
    # Now we will plot our curve on our frame
    mean_polynomial1 = y_polynomial1.mean()
    
    fig, ax3 = plt.subplots(figsize=(8,5),dpi=100)
    ax3.imshow(data_full)
    ax3.plot(xp1, y_polynomial1, color="orange", label=f"Polynomial")
    ax3.axhline(mean_polynomial1, label="Polinomial mean")
    ax3.set_title("ax3")
    ax3.legend()
    
    delta_y1 = []
    for i in range(0,len(data_full[0])):
        delta1 = mean_polynomial1 - polynomial1(xp1)[i]
        delta_y1.append(rounding(delta1))
    
    # Транспонирование матриц для работы со столбцами
    data_columns1 = np.transpose(data_part).tolist()
    data_collumns_entire_image1 = np.transpose(data_full).tolist()

    # Нахождение минимальных значений и их индексов
    data_col_min1 = np.min(data_part, axis=0)
    index1 = np.argmin(data_part, axis=0).tolist()

    shape1 = np.array(data_collumns_entire_image1).shape
    new_pic_col1 = np.zeros(shape1,dtype="uint16")
    for i in range(0,len(data_collumns_entire_image1)):
        for j in range(0,len(data_collumns_entire_image1[i])):
            m  = j + delta_y1[i]
            if m<len(data_collumns_entire_image1[i]):
                new_pic_col1[i][m] = data_collumns_entire_image1[i][j] 
            else:
                m = m - len(data_collumns_entire_image1[i])
                new_pic_col1[i][m] = data_collumns_entire_image1[i][j]  
    
    data_rows1 = []
    for i in range(0,len(new_pic_col1[0])):
        data_rows1.append(list(new_pic_col1[:,i]))
            
    
    delta_x1 = np.arange(0,len(data_rows1))*tg1
    shape1 = np.array(data_rows1).shape
    new_data_rows1 = np.zeros(shape1,dtype="uint16")
    for i in range(0,len(data_rows1)):
        for j in range(0,len(data_rows1[i])):
            m  = rounding(j + 2*delta_x1[i])
            if m<len(data_rows1[i]):
                new_data_rows1[i][m] = data_rows1[i][j] 
            else:
                m = m - len(data_rows1[i])
                new_data_rows1[i][m] = data_rows1[i][j] 

    aligned = new_data_rows1

    fig, ax4 = plt.subplots(figsize=(8,5),dpi=100)
    ax4.imshow(aligned, cmap='gray')
    ax4.grid()
    ax4.set_title("Aligned image")

    return aligned


def polynomial_align_spectra(data,low_limit,up_limit):
    
    """
    function aligns the spectrum through finding min value in each column and approximate it with polynomial and mean for polinomial line
    
    data - whole image
    up_limit,low_limit - limits of spectra in rows number
    
    """
    
    data_part = data[low_limit-10:up_limit+10]
    fig, ax1 = plt.subplots(figsize=(8,5),dpi=100)
    ax1.imshow(data_part, cmap='gray')
#     plt.imsave('pic_lamp_20s.jpg', data_part)
    ax1.set_title("Initial part of spectra")
#     plt.savefig("initial_part.eps")
    
    #create massive of part of whole spectra with arrays of columns, not rows
    data_columns = []
    for i in range(0,len(data_part[0])):
        data_columns.append(list(data_part[:,i]))
    data_columns=list(data_columns)
    
    #create massive of whole spectra with arrays of columns, not rows
    data_collumns_entire_image = []
    for j in range(0,len(data[0])):
        data_collumns_entire_image.append(list(data[:,j]))
    data_collumns_entire_image=list(data_collumns_entire_image)

    #find min value in each column and then find indexes of the values - they will be Y-coordinate
    data_col_min = data_part.min(axis=0)

#     data_col_min = data_part.max(axis=0)
    index = []
    for i in range(0,len(data_col_min)):
            index.append(data_columns[i].index(data_col_min[i]))


    #find median value of neighboring indexes of min values in each column in some ranges to compairing in the next steps
    #we schould find mediam values for ranges not for whole image, because we have non-linear graph and it has curvature  
    median = []
    n = 0
    while n+100<len(index):
        k=n+100
        median.append([n,k,np.min(index[n:k])])
        n=k
    else:
        median.append([n,-1,np.min(index[n:])])

    #calculate mean delta between neighboring values
    delt=[]
    for i in range(1,len(index)):
        delt.append(abs(index[i]-index[i-1]))
    delt_mean = np.array(delt).sum()/len(delt)    

    #now we compair difference between median value indexes of minimum value in each column     
    for i in median:
        for j in range(i[0],i[1]):
            if abs(i[2]-index[j])>delt_mean:
                index[j] = i[2]

    fig, ax2 = plt.subplots(figsize=(8,5),dpi=100)
    ax2.plot(index)
    
    xp = np.arange(0, len(index))
    X = np.array(xp).reshape(-1, 1)  # Преобразуем xp в двумерный массив для совместимости с sklearn

    best_degree = None
    best_score = float('inf')
    best_polynomial = None
    best_y = None

    # Тестируем полиномы разной степени
    for degree in range(1, 6):  # Диапазон степеней от 1 до 5
        # Создаем модель RANSAC с полиномиальной регрессией текущей степени
        poly_model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
        ransac = RANSACRegressor(base_estimator=poly_model,
                                 min_samples=int(len(xp) * 0.8),
                                 residual_threshold=2,
                                 random_state=42)

        # Обучаем модель
        ransac.fit(X, index)

        # Оцениваем качество аппроксимации
        y_pred = ransac.predict(X)
        score = mean_squared_error(index, y_pred)

        # Обновляем лучшую модель, если найдена более подходящая степень
        if score < best_score:
            best_score = score
            best_degree = degree
            best_polynomial = np.poly1d(np.polyfit(xp, index, degree))
            best_y = y_pred

    y_polynomial = best_y
    
    polynomial = best_polynomial
    
    ax2.plot(index, color="steelblue", label="Медианное значение") 
    ax2.plot(xp, y_polynomial, color="orange")
#     ax2.set_title("Polinomial approximation of spectra")
    ax2.set_xlabel("Номер столбца пикселей")
    ax2.set_ylabel("Номер строки пикселей")
    ax2.legend()
    #plt.savefig("polinomial_spectra.jpg")
    
    xp_tg = np.arange(0,len(y_polynomial))
    z_tg = np.polyfit(xp_tg,y_polynomial, 1)
    polynomial_tg = np.poly1d(z_tg)
    y_polynomial_tg = polynomial_tg(xp_tg)
    tg = polynomial_tg[1]
    
    #now we will plot our curve on our frame
    mean_polynomial = y_polynomial.mean()
    fig, ax3 = plt.subplots(figsize=(8,5),dpi=100)
    ax3.imshow(data_part, cmap='gray')
    ax3.plot(xp, y_polynomial, color="orange")
    ax3.axhline(mean_polynomial, label="Polinomial mean")
    ax3.set_axis_off()
#     ax3.legend(bbox_to_anchor=(1.01,0.5))
    #plt.savefig("polinomial_mean.jpg")
    
    delta_y = []
    for i in range(0,len(index)):
        delta = mean_polynomial - polynomial(xp)[i] #initial
#         delta = mean_polynomial + polynomial(xp)[i] 
        
        delta_y.append(int(delta))

    shape = np.array(data_collumns_entire_image).shape
    new_pic_col = np.zeros(shape)
    for i in range(0,len(data_collumns_entire_image)):
        for j in range(0,len(data_collumns_entire_image[i])):
            m  = j + delta_y[i] #initial
#             m  = j - delta_y[i]
            
            if m<len(data_collumns_entire_image[i]):#### initial <
                new_pic_col[i][m] = data_collumns_entire_image[i][j] 
            else:
                m = m - len(data_collumns_entire_image[i])
                new_pic_col[i][m] = data_collumns_entire_image[i][j] 
    data_rows = []
    for i in range(0,len(new_pic_col[0])):
        data_rows.append(list(new_pic_col[:,i]))
            
    
    delta_x = np.arange(0,len(data_rows))*tg
    shape = np.array(data_rows).shape
    new_data_rows = np.zeros(shape)
    for i in range(0,len(data_rows)):
        for j in range(0,len(data_rows[i])):
            m  = int(j + 2*delta_x[i])
            if m<len(data_rows[i]):
                new_data_rows[i][m] = data_rows[i][j] 
            else:
                m = m - len(data_rows[i])
                new_data_rows[i][m] = data_rows[i][j] 
    
    aligned_image = new_data_rows
    
    fig, ax4 = plt.subplots(figsize=(8,5),dpi=100)
    ax4.imshow(aligned_image, cmap='gray')
    plt.axis("off")
    # ax4.grid()
#     ax4.set_title("Aligned spectra")
    #plt.savefig("aligned_spectra.jpg")
    
    return aligned_image




def get_indexis_from_lamp(clean_data):
# Инвертируем массив (0 становятся 1, и наоборот)


    #clean_data = data_mask
    data_mask = np.where(clean_data == 0.00001, 0, 1).astype(np.uint8)  # Убедитесь, что data_mask имеет тип uint8
    clean_data = data_mask   
    
    inverted_array_part = np.where(data_mask == 0, 255, 0).astype(np.uint8)
    inverted_array_full = np.where(clean_data == 0, 255, 0).astype(np.uint8)

    # Нет необходимости преобразовывать data_mask в градации серого, так как оно уже в подходящем формате
    contours_part, _ = cv2.findContours(inverted_array_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Создание цветного изображения для визуализации уже не требует преобразования цветов
    
#     # Создаем цветное изображение для визуализации
    color_image_part = cv2.cvtColor(data_mask, cv2.COLOR_GRAY2BGR)
#     # Центры фигур и их контуры
    centers_part = []

    # Вычисляем размеры каждого контура (используем площадь ограничивающего прямоугольника)
    contour_sizes = [cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3] for contour in contours_part]

    # Вычисляем средний размер контуров
    average_size = np.mean(contour_sizes)/2

    # Фильтруем контуры, оставляя только те, чей размер больше или равен среднему
    filtered_contours = [contour for contour, size in zip(contours_part, contour_sizes) if size >= average_size]

    contours_part = filtered_contours
    
    for contour in contours_part:
        # Вычисляем центр контура
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers_part.append((cX, cY))

            # Рисуем контур и центр на изображении
            cv2.drawContours(color_image_part, [contour], -1, (0, 255, 0), 2)
            cv2.circle(color_image_part, (cX, cY), 5, (255, 0, 0), -1)
            
    plt.figure(figsize=(10,6))

    plt.imshow(color_image_part)
    plt.title("color_image_part")
            
    x_coordinates = [center[0] for center in centers_part]
    y_coordinates = [center[1] for center in centers_part]


    plt.show()

    return x_coordinates, y_coordinates


def full_reduction(file):
    
    fit = fits.open(file)
    data = fit[0].data
    mask = np.where(data < np.mean(data)*.1, 1,0) 
    plt.figure(figsize=(10,6))
    plt.imshow(mask)
    plt.title("data_mask")  

    inverted_array_full = np.where(mask == 0, 0, 255).astype(np.uint8)

    # Находим контуры
    contours, _ = cv2.findContours(inverted_array_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Создаем цветное изображение для визуализации
    color_image = cv2.cvtColor(inverted_array_full, cv2.COLOR_GRAY2BGR)
                               
    vertical_contours = []
    quarter_height = data.shape[0] / 3  # Четверть высоты изображения

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Проверяем, занимает ли контур больше четверти высоты изображения
        if h/w > 2:# quarter_height:
            vertical_contours.append(contour)
            # Закрашиваем контур в изображении
            cv2.drawContours(color_image, [contour], -1, 0, thickness=cv2.FILLED)  # Закрашиваем красным для наглядности


    inverted_array_full = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(10,6))
    plt.imshow(inverted_array_full)
    plt.title("data_mask_without_vlines")   


        # Находим контуры
    contours, _ = cv2.findContours(inverted_array_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Создаем цветное изображение для визуализации
    color_image = cv2.cvtColor(inverted_array_full, cv2.COLOR_GRAY2BGR)

      # Анализ размеров контуров
    width = []
    height = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        width.append(w)
        height.append(h)

    # Вычисление средних значений ширины и высоты
    width_mean = np.mean(width)
    height_median = np.mean(height)

    # Параметры для разделения изображения на прямоугольные области
    cols, rows = 10, 10  # Разделение изображения на 4x4 области
    rect_width, rect_height = inverted_array_full.shape[1] // cols, inverted_array_full.shape[0] // rows
    # Рисуем сетку прямоугольников на цветном изображении

    # Создаем цветное изображение для визуализации
    color_image = cv2.cvtColor(inverted_array_full, cv2.COLOR_GRAY2BGR)
    region_index_1 = []
    # Разделение изображения на области и определение принадлежности контуров
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width_mean and h > color_image.shape[0] * 0.05:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Определение принадлежности к области
                col_index = cX // rect_width
                row_index = cY // rect_height
                region_index_1.append(row_index * cols + col_index)
                # Рисуем контур и центр на изображении
                cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(color_image, (cX, cY), 5, (0, 255, 0), -1)

                for row in range(rows):
                    for col in range(cols):
                        cv2.rectangle(color_image, (col * rect_width, row * rect_height), ((col + 1) * rect_width, (row + 1) * rect_height), (255, 255, 0), 2)
                        region_index = row * cols + col
                        text_pos = (col * rect_width , row * rect_height +200)
                        # Рисуем индекс области
                        cv2.putText(color_image, str(region_index), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

    region_index_1_sorted = np.sort(region_index_1)
    # Инициализация нового списка строк
    region_index_1_sorted_str = []

    # Преобразование каждого элемента в строку с учетом ведущего нуля
    for i in region_index_1_sorted:
        if i < 10:
            region_index_1_sorted_str.append(f'0{i}')
        else:
            region_index_1_sorted_str.append(f'{i}')

    first_num = []
    second_num = []

    for i in range(len(region_index_1_sorted_str)):

        first_num.append(region_index_1_sorted_str[i][0])
        second_num.append(region_index_1_sorted_str[i][1])

     # Подсчет количества повторений для каждого значения
    first_count = Counter(first_num)
    second_count = Counter(second_num)

    # Преобразование Counter в словарь
    first_result_dict = dict(first_count)   
    second_result_dict = dict(second_count) 

    plt.figure(figsize=(10, 6))

    if len(first_result_dict) > len(second_result_dict):
        print("vertical")
        second_result_dict_values_u =  np.array([int(key) for key, value in second_result_dict.items() if int(key)>=7])
        if len(second_result_dict_values_u) > 0:
            second_result_dict_values_u = second_result_dict_values_u.min()
        else:
            second_result_dict_values_u = rect_width*10
        second_result_dict_values_l =  np.array([int(key) for key, value in second_result_dict.items() if int(key)<2])
        if len(second_result_dict_values_l) > 0:
            second_result_dict_values_l = second_result_dict_values_l.max()

        else:
            second_result_dict_values_l = -1

        plt.axvline(rect_width*(second_result_dict_values_u),lw=5)

        plt.axvline(rect_width*(second_result_dict_values_l+1),lw=5)
        data_cropped = data[:,int(rect_width*(second_result_dict_values_l+1)):int(rect_width*(second_result_dict_values_u))]

    elif len(first_result_dict) < len(second_result_dict):
        first_result_dict_values_u =  np.array([int(key) for key, value in first_result_dict.items() if int(key)>=2])
        if len(first_result_dict_values_u) > 0:
            first_result_dict_values_u = first_result_dict_values_u.min()
        else:
            first_result_dict_values_u = rect_height*10
        first_result_dict_values_l =  np.array([int(key) for key, value in first_result_dict.items() if int(key)<7])
        if len(first_result_dict_values_l) > 0:
            first_result_dict_values_l = first_result_dict_values_l.max()

        else:
            first_result_dict_values_l = -1
        print("horizontal") 

        plt.axhline(rect_height*(first_result_dict_values_u-1),lw=5)

        plt.axhline(rect_height*(first_result_dict_values_l+1),lw=5)
        if first_result_dict_values_l+2 < 2: #чтобы не брал слишком много
            data_cropped = data[int(rect_height*(first_result_dict_values_l+2)):int(rect_height*(first_result_dict_values_u-1))]
        else:
            data_cropped = data[int(rect_height*(first_result_dict_values_l+1)):int(rect_height*(first_result_dict_values_u-1))]

    elif len(first_result_dict)== 1  & len(second_result_dict) == 1:
        print("check location") 
        data_cropped = data
    else:    
        print("help")
        data_cropped = data
                               
    # Визуализация результата
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    plt.title("Processed Image with Contours and Regions")
    plt.show()
                               

    data_rotated = rotate_spectra(data_cropped)
    plt.figure(figsize=(10, 6))
    plt.imshow(data_rotated)

    mask = np.where(data_rotated < np.mean(data_rotated)*.1, 1,0)#.00001)  # Предполагается, что переменная `value` задана где-то выше
    inverted_array_full_1 = np.where(mask == 0, 0, 255).astype(np.uint8)
   

      # Находим контуры
    color_image = cv2.cvtColor(inverted_array_full_1, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertical_contours = []
    quarter_height = inverted_array_full_1.shape[0] / 3  # Четверть высоты изображения

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Проверяем, занимает ли контур больше четверти высоты изображения
        if h/w > 2:# quarter_height:
            vertical_contours.append(contour)
            # Закрашиваем контур в изображении
            cv2.drawContours(color_image, [contour], -1, 0, thickness=cv2.FILLED)  # Закрашиваем красным для наглядности

    plt.figure(figsize=(10,6))
    plt.imshow(inverted_array_full_1)
    plt.title("data_mask_rot_crop")   


    # Инициализируем минимальные и максимальные координаты
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    # Проходим по всем контурам
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x, min_y = min(min_x, x), min(min_y, y)
        max_x, max_y = max(max_x, x+w), max(max_y, y+h)

    # Рисуем минимальный общий прямоугольник на изображении
    cv2.rectangle(color_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Используем Matplotlib для отображения изображения
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))  # Конвертируем цвета из BGR в RGB для корректного отображения
    plt.title('Min Enclosing Rectangle')
    plt.show()


    padding = 100
    padded_min_x = max(min_x - padding, 0)
    padded_min_y = max(min_y - padding, 0)
    padded_max_x = min(max_x + padding, color_image.shape[1])
    padded_max_y = min(max_y + padding, color_image.shape[0])

    # Обрезаем изображение
    cropped_image = data_rotated[padded_min_y:padded_max_y, padded_min_x:padded_max_x]
    
    mask_aaa = np.where(cropped_image < np.mean(cropped_image)*.1, 1,0)#.00001)  # Предполагается, что переменная `value` задана где-то выше

    cropped_mask  = inverted_array_full_1[padded_min_y:padded_max_y, padded_min_x:padded_max_x]

    # Используем Matplotlib для отображения обрезанного изображения
    plt.figure(figsize=(10, 10))
    plt.imshow(cropped_mask)  # Конвертируем цвета из BGR в RGB для корректного отображения
    plt.title('Cropped Image with Padding')
    plt.axis('off')  # Скрываем оси координат
    plt.show()
      # Находим контуры
    color_image_aaa = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(color_image_aaa, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vertical_contours = []
    quarter_height = cropped_mask.shape[0] / 3  # Четверть высоты изображения

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Проверяем, занимает ли контур больше четверти высоты изображения
        if h/w > 2:# quarter_height:
            vertical_contours.append(contour)
            # Закрашиваем контур в изображении
            cv2.drawContours(color_image_aaa, [contour], -1, 0, thickness=cv2.FILLED)  # Закрашиваем красным для наглядности
    gray_aaa = cv2.cvtColor(color_image_aaa, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(10,6))
    plt.imshow(color_image_aaa)
    plt.title("data_mask_rot_crop") 
    
    aligned_image = align_20240118_weight_mask(cropped_image, gray_aaa)

    
    return aligned_image,data




def calib_image(data,root,best_poly,polynomial_extrapolate):
    data_image = np.array(np.log10(65535/data)*1000)
    # Векторизуем функцию best_poly, чтобы она могла принимать массивы NumPy
    vectorized_best_poly = np.vectorize(best_poly)
    vectorized_extrapolate = np.vectorize(polynomial_extrapolate)
    # Применяем функцию best_poly к элементам массива, которые больше root
    data_calib = np.where(data_image < root, vectorized_extrapolate(data_image), 
                          vectorized_best_poly(data_image))
    plt.imshow(data_calib)
    return data_calib



def get_values_with_difference(arr):
    result = []
    arr = np.append(arr,arr[-1]+10)
    for i in range(len(arr) - 1):
        if arr[i+1] - arr[i] > 1:
            result.append(arr[i])
    
    return result


def rounding(value):

    try:
        value = float(value)
    except ValueError:
        print("Input should be a digit or a float!")

    if value > 0:
        sign = 1
    else:
        sign = -1

    value = abs(value)

    frac = value - int(value)

    if frac >= 0.5:
        return math.ceil(value)*sign
    else:
        return math.floor(value)*sign

    
    
    
    
def polynomial_align_universal_mse5(data,up,low, order_mse=None):
    
    sum_intens_row1 = preob_spectra(data)  #preob_spectra  function1
    x_indexes = np.arange(len(sum_intens_row1))
    
    band_end = get_band_end6(sum_intens_row1) #get_band_end
    cross_points_ind_1 = get_cross_points_ind(sum_intens_row1, band_end) #get_cross_points_ind
    # print('points',cross_points_ind_1)
    x1 = sum_intens_row1[cross_points_ind_1[0]:cross_points_ind_1[1]]
    peaks1, _ = find_peaks(x1,distance = 10)  #find_peaks    CHANGED
    # print("PEAKS ",peaks1)
#     plt.figure(figsize=(10,6))
#     plt.plot(sum_intens_row1,c="y")
#     plt.plot(np.arange(len(x1))+cross_points_ind_1[0],x1)
#     plt.scatter(peaks1+cross_points_ind_1[0],x1[peaks1])
    
#     fig,ax0 = plt.subplots(figsize=(8,5),dpi=100)
# #     print(cross_points_ind_1[0],cross_points_ind_1[-1])
#     ax0.plot(x_indexes[cross_points_ind_1[0]:cross_points_ind_1[-1]],sum_intens_row1[cross_points_ind_1[0]:cross_points_ind_1[-1]])
#     ax0.plot(x_indexes,sum_intens_row1,color="red")
#     ax0.set_title("aaaaaaaaaaaaaaaaaaaa")
    
    results_full1 = peak_widths(x1, peaks1, rel_height=0.8) #peak_widths
    results_half = peak_widths(x1, peaks1, rel_height=0.5) #peak_widths
    
#     print(results_half)
    
    data_part1 = data[up:low]# data[int(cross_points_ind_1[0]*0.9):int(cross_points_ind_1[-1]*1.1)]    
    
    # fig,ax1 = plt.subplots(figsize=(8,5),dpi=100)

#     ax1.plot(x_indexes[cross_points_ind_1[0]:cross_points_ind_1[-1]],sum_intens_row1[cross_points_ind_1[0]:cross_points_ind_1[-1]])
#     ax1.set_title("The lowest band")
#     plt.scatter(peaks1+cross_points_ind_1[0],sum_intens_row1[peaks1+cross_points_ind_1[0]])
#     ax1.axhline(band_end[0], c='red')
#     ax1.axhline(band_end[1], c='green')
#     plt.hlines(*results_half[1:]+cross_points_ind_1[0], color="C4")
#     plt.hlines(*results_full1[1:]+cross_points_ind_1[0], color="C8")
    
    fig,ax2 = plt.subplots(figsize=(8,5),dpi=100)
    
    ax2.imshow(data_part1, cmap='gray')    
    
    #_____________________align_first______________________________________
    
      #create massive of part of whole spectra with arrays of columns, not rows
    data_columns1 = []
    for i in range(0,len(data_part1[0])):
        data_columns1.append(list(data_part1[:,i]))
    data_columns1=list(data_columns1)
    
    #create massive of whole spectra with arrays of columns, not rows
    data_collumns_entire_image1 = []
    for j in range(0,len(data[0])):
        data_collumns_entire_image1.append(list(data[:,j]))
    data_collumns_entire_image1=list(data_collumns_entire_image1)

    #find min value in each column and then find indexes of the values - they will be Y-coordinate
    data_col_min1 = []
    for i in range(0,len(data_columns1)):
        data_col_min1.append(np.min(data_columns1[i]))
        
#     index1 = data_col_min1
        
    index1 = []
    for i in range(0,len(data_col_min1)):
            index1.append(data_columns1[i].index(data_col_min1[i]))
    #find median value of neighboring indexes of min values in each column in some ranges to compairing in the next steps
    #we schould find mediam values for ranges not for whole image, because we have non-linear graph and it has curvature  
    median1 = []
    n = 0
    while n+300<len(index1):
        k=n+300
        median1.append([n,k,np.median(index1[n:k])])
#         median1.append([n,k,np.mean(index1[n:k])])
        
        n=k
    else:
        median1.append([n,-1,np.median(index1[n:])])
#         median1.append([n,-1,np.mean(index1[n:])])

    #calculate mean delta between neighboring values
    delt1=[]
    for i in range(1,len(index1)):
        delt1.append(abs(index1[i]-index1[i-1]))
    delt_mean1 = np.array(delt1).sum()/len(delt1)    

    #now we compair difference between median value indexes of minimum value in each column     
    for i in median1:
        for j in range(i[0],i[1]):
            if abs(i[2]-index1[j])>delt_mean1:
                index1[j] = i[2]

#     median_y_pol = np.median(index1)
#     for i,y in enumerate(index1):
#         if y<median_y_pol:
#             index1[i]=median_y_pol
    
    #plot polynomial
    xp1 = np.arange(0,len(index1))
    
#########################################################################
###################################mse###################################
#########################################################################
#########################################################################
#     print('xp1',xp1)
    order_mse = order_mse
    len_x = len(xp1)
#     print("len_x", len_x)
    previous_mse = None  # Initialize as None

    while True:
        z1, residuals, *_ = np.polyfit(xp1, index1, order_mse, full=True)
        
        mse = residuals[0] / len_x
        print(order_mse, mse)
        if len(residuals) == 0:
            break
        if previous_mse is not None and mse != 0:
            change_percentage = abs((mse - previous_mse) / previous_mse)
            if change_percentage <= 0.01:  # Check if change is <= 1%
                order_mse = order_mse - 1
                break
                
        order_mse += 1
        previous_mse = mse
        
    # print("ORDER", order_mse)
    polynomial1 = np.poly1d(z1)
    
    
    y_polynomial1 = polynomial1(xp1)

#########################################################################
#########################################################################
#################################mse#####################################
#########################################################################

    fig, ax3 = plt.subplots(figsize=(8,5),dpi=100)
#     ax3.plot(index1)
    ax3.plot(index1, color="steelblue", label="Spectra line") 
    ax3.plot(xp1, y_polynomial1, color="orange", label=f"Polynomial, order_mse = {order_mse}")
#     ax3.set_title("Polinomial approximation of spectra")
    ax3.set_title("ax3")
    ax3.legend()
    
    xp_tg1 = np.arange(0,len(y_polynomial1))
    z_tg1 = np.polyfit(xp_tg1,y_polynomial1, 1)
    polynomial_tg1 = np.poly1d(z_tg1)
    y_polynomial_tg1 = polynomial_tg1(xp_tg1)
    tg1 = polynomial_tg1[1]
    
    #now we will plot our curve on our frame
    mean_polynomial1 = y_polynomial1.mean()
    fig, ax4 = plt.subplots(figsize=(8,5),dpi=100)
    ax4.imshow(data_part1, cmap='gray')
    ax4.plot(xp1, y_polynomial1, color="orange", label=f"Polinomial, order = {order_mse}")
    ax4.axhline(mean_polynomial1, label="Polinomial mean")
    ax4.set_title("ax4")
    
    delta_y1 = []
    for i in range(0,len(index1)):
        delta1 = mean_polynomial1 - polynomial1(xp1)[i]
        delta_y1.append(rounding(delta1))

    shape1 = np.array(data_collumns_entire_image1).shape
    new_pic_col1 = np.zeros(shape1,dtype="uint16")
    # print("sssshape!!!!!!!!!!!!!1",shape1)
    # print("ZEROES ",new_pic_col1.dtype)
    for i in range(0,len(data_collumns_entire_image1)):
        for j in range(0,len(data_collumns_entire_image1[i])):
            m  = j + delta_y1[i]
            if m<len(data_collumns_entire_image1[i]):
                new_pic_col1[i][m] = data_collumns_entire_image1[i][j] 
            else:
                m = m - len(data_collumns_entire_image1[i])
                new_pic_col1[i][m] = data_collumns_entire_image1[i][j]  
    
    data_rows1 = []
    for i in range(0,len(new_pic_col1[0])):
        data_rows1.append(list(new_pic_col1[:,i]))
            
    
    delta_x1 = np.arange(0,len(data_rows1))*tg1
    shape1 = np.array(data_rows1).shape
    new_data_rows1 = np.zeros(shape1,dtype="uint16")
    for i in range(0,len(data_rows1)):
        for j in range(0,len(data_rows1[i])):
            m  = rounding(j + 2*delta_x1[i])
            if m<len(data_rows1[i]):
                new_data_rows1[i][m] = data_rows1[i][j] 
            else:
                m = m - len(data_rows1[i])
                new_data_rows1[i][m] = data_rows1[i][j] 
    
    aligned_image1 = new_data_rows1
 

    
    aligned = new_data_rows1#new_data_rotated_rows
    
    
#     ############################################################
    fig, ax13 = plt.subplots(figsize=(8,5),dpi=100)
    ax13.imshow(aligned, cmap='gray')
    ax13.grid()
    ax13.set_title("ax13")
    
    
#     print("AFTER CV2 ",data_rotated.dtype )
    

    return aligned, polynomial1





def date_obs(sid_h,S1,grin_mid):
    """
    sid_h -- sidereal time of D2 (second date) 
    S1 -- local sidereal time (from obs log)
    gin_mid -- Time(f"{D1} 18:00:00")
    """
    
    date_obs_one=""
    date_obs_jd=""
    
    if sid_h.value>=0 and sid_h.value<=12:

        if S1>=12 and S1<24: 
            delta = sid_h.value - S1  
            if abs(delta) > 8: #before mid
                delta = delta%24
                date = grin_mid - delta*u.hour
                date_obs_one = date.fits
                date_obs_jd = date.jd
            else: #after mid
                date = grin_mid - delta*u.hour
                date_obs_one = date.fits
                date_obs_jd = date.jd

        elif S1>=0 and S1<=12:
            delta = sid_h.value - S1
            date = grin_mid - delta*u.hour
            date_obs_one = date.fits
            date_obs_jd = date.jd

    elif sid_h.value>=12 and sid_h.value<24:

        if S1>=12 and S1<24:
            delta = sid_h.value - S1  
            date = grin_mid - delta*u.hour
            date_obs_one = date.fits
            date_obs_jd = date.jd

        elif S1>=0 and S1<=12:
            delta = S1 - sid_h.value
            if abs(delta) > 8: #after mid
                delta = delta%24
                date = grin_mid + delta*u.hour
                date_obs_one = date.fits
                date_obs_jd = date.jd
            else:  #before mid
                date = grin_mid + delta*u.hour
                date_obs_one = date.fits
                date_obs_jd = date.jd
            
    return date_obs_one, date_obs_jd

def two_month(D):
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[1][2]}-{D[0][1]}-{D[0][0]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}"
    D2_Time_func = Time(f"{D2} 00:00:00") #date2 in Time format
    sid_h = ((D2_Time_func.sidereal_time('mean',longitude=longitude).value - 6)%24)*u.hour
    grin_mid = Time(f"{D1} 18:00:00")
    return sid_h, grin_mid

def one_month(D):
    D=D.split("-")
    D[1] = D[1].split(".")  
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[1][2]}-{D[1][1]}-{D[0]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}"
    D2_Time_func = Time(f"{D2} 00:00:00") #date2 in Time format
    sid_h = ((D2_Time_func.sidereal_time('mean',longitude=longitude).value - 6)%24)*u.hour
    grin_mid = Time(f"{D1} 18:00:00")
    return sid_h, grin_mid

def universal_time_one_month(D,T):
    """
    Function calculates and return DATE-OBS if obs was done in the same month (D) and we have time of obs (T) in UT
    """
    D=D.split("-")
    D[1] = D[1].split(".")  
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[1][2]}-{D[1][1]}-{D[0]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}"
    T_hms=T.split(":")
    if len(T_hms[2]) >1:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            day_obs = Time(f"{D1} {T}")
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            day_obs = Time(f"{D2} {T}")
    else:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            day_obs = Time(f"{D1} {T}00")
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            day_obs = Time(f"{D2} {T}00")
        
    return day_obs
    
def universal_time_two_month(D,T):
    """
    Function calculates and return DATE-OBS if obs was done in two month (D) and we have time of obs (T) in UT
    30.09-01.10.1956
    """
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[1][2]}-{D[0][1]}-{D[0][0]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}"
    T_hms=T.split(":")
    if len(T_hms[2]) >1:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            day_obs = Time(f"{D1} {T}")
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            day_obs = Time(f"{D2} {T}")
    else:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            day_obs = Time(f"{D1} {T}00")
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            day_obs = Time(f"{D2} {T}00")
        
    return day_obs

def universal_time_two_years(D,T):
    """
    31.12.1956-01.01.1957
    """
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[0][2]) < 4:#if we have '96' instead of '1996'
        D[0][2] = f"19{D[0][2]}"
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[0][2]}-{D[0][1]}-{D[0][0]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}"
    T_hms=T.split(":")
    if len(T_hms[2]) >1:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            day_obs = Time(f"{D1} {T}")
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            day_obs = Time(f"{D2} {T}")
    else:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            day_obs = Time(f"{D1} {T}00")
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            day_obs = Time(f"{D2} {T}00")    
    return day_obs



def time_format(i,time_tab):
    """
    time_tab is an array from table
    time_table is the first element (start of observation / time) from time_tab
    function returns time in format hh:mm:ss
    """
    
    time_1 = time_tab
    
    try:
        if "s" in time_1 and "m" in time_1 and "h" in time_1:
            time_1 = time_1.replace('s','')
            time_1 = time_1.split('h')
            time_m = time_1[1].split('m')
            time_1 = f"{time_1[0]}:{time_m[0]}:{time_m[1]}"

        elif "s" not in time_1 and "m" in time_1 and "h" in time_1:
            time_1 = time_1.split('h')
            if ".m" in time_1[1] or "m." in time_1[1]:
                time_m = time_1[1].replace('m','')
                minutes = str(float(time_m[0])).split(".")
                # if int(minutes[1]) > 0:    
                if len(str(time_1[0])) > 1 and len(str(minutes[0])) > 1 and len(str(minutes[1]*6)) > 1:
                    time_1 = f"{time_1[0]}:{minutes[0]}:{minutes[1]*6}"
                elif len(str(time_1[0])) > 1 and len(str(minutes[0])) > 1 and len(str(minutes[1]*6)) <= 1:
                    time_1 = f"{time_1[0]}:{minutes[0]}:0{minutes[1]*6}"
                elif len(str(time_1[0])) > 1 and len(str(minutes[0])) <= 1 and len(str(minutes[1]*6)) <= 1:
                    time_1 = f"{time_1[0]}:0{minutes[0]}:0{minutes[1]*6}"
                elif len(str(time_1[0])) <= 1 and len(str(minutes[0])) > 1 and len(str(minutes[1]*6)) <= 1:
                    time_1 = f"0{time_1[0]}:{minutes[0]}:0{minutes[1]*6}"
                elif len(str(time_1[0])) <= 1 and len(str(minutes[0])) <= 1 and len(str(minutes[1]*6)) <= 1:
                    time_1 = f"0{time_1[0]}:0{minutes[0]}:0{minutes[1]*6}"
                elif len(str(time_1[0])) <= 1 and len(str(minutes[0])) <= 1 and len(str(minutes[1]*6)) > 1:
                    time_1 = f"0{time_1[0]}:0{minutes[0]}:{minutes[1]*6}"
                elif len(str(time_1[0])) > 1 and len(str(minutes[0])) <= 1 and len(str(minutes[1]*6)) > 1:
                    time_1 = f"{time_1[0]}:0{minutes[0]}:{minutes[1]*6}"
                elif len(str(time_1[0])) <= 1 and len(str(minutes[0])) > 1 and len(str(minutes[1]*6)) > 1:
                    time_1 = f"0{time_1[0]}:{minutes[0]}:{minutes[1]*6}"

                # else:
                #     time_1 = f"{time_1[0]}:{minutes[0]}:00"
            else:
                time_m = time_1[1].split('m')
                if len(time_m)>1:
                    if len(str(time_m[1]))>1: 
                        time_1 = f"{time_1[0]}:{time_m[0]}:{time_m[1]}"
                    else:
                        time_1 = f"{time_1[0]}:{time_m[0]}:0{time_m[1]}"

                else:
                    minutes = str(float(time_m[0])).split(".")
                    if int(minutes[1]) > 0:
                        if len(str(minutes[1]*6)) > 1:
                            time_1 = f"{time_1[0]}:{minutes[0]}:{minutes[1]*6}"
                        else:
                            time_1 = f"{time_1[0]}:{minutes[0]}:0{minutes[1]*6}"
                    else:
                        time_1 = f"{time_1[0]}:{minutes[0]}:00"

        elif "s" not in time_1 and "m" not in time_1 and "h" in time_1:
            time_1 = time_1.split('h')
            minutes = str(float(time_1[1])).split(".")
            if int(minutes[1]) > 0:
                if len(str(minutes[1]*6)) > 1:
                    time_1 = f"{time_1[0]}:{minutes[0]}:{minutes[1]*6}"
                else:
                    time_1 = f"{time_1[0]}:{minutes[0]}:0{minutes[1]*6}"

            else:
                time_1 = f"{time_1[0]}:{minutes[0]}:00"
    except ValueError:
        print(i)
            
    time_check = time_1.split(":")
    if len(time_check[0]) <=1:
        time_1 = f"0{time_check[0]}:{time_check[1]}:{time_check[2]}"
    if len(time_check[1]) <=1:
        time_1 = f"{time_check[0]}:0{time_check[1]}:{time_check[2]}"  
    if len(time_check[2]) <=1:
        time_1 = f"{time_check[0]}:{time_check[1]}:0{time_check[2]}"
    
    time_1 = time_1.replace(" ","")
    return time_1


def decret_time_one_month(D,T):
    """
    D - DATE_OBS from table
    T - time of start of observ from function time_format(time_tab) in hh:mm:ss format
    Return separately year, month, day, full date (time function), 
    known sunday (time function), time (hh:mm:ss) and array from time (hh, mm, ss)
    """
    try:
        if "-" in D:#02-03.03.1999
            D=D.split("-")
            D[1] = D[1].split(".")  
            if len(D[1][2]) < 4:#if we have '96' instead of '1996'
                D[1][2] = f"19{D[1][2]}"
            D1 = f"{D[1][2]}-{D[1][1]}-{D[0]}".replace(" ","")
            D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}".replace(" ","")

            y = int(D[1][2])
            m = int(D[1][1])
            sunday = Time(f'2022-05-01 {T}')

            T_hms=T.split(":")
            if int(T_hms[0]) >= 0  and int(T_hms[0]) < 12:
                d = int(D[1][0])
                date = Time(f'{D2} {T}')
            elif int(T_hms[0]) >= 12  and int(T_hms[0]) < 24:
                d = int(D[0])
                date = Time(f'{D1} {T}') 

        else:#02.03.1988
            D = D.split(".")  
            if len(D[2]) < 4:#if we have '96' instead of '1996'
                D[2] = f"19{D[2]}"
            D1 = f"{D[2]}-{D[1]}-{D[0]}".replace(" ","")
            y = int(D[2])
            m = int(D[1])
            d = int(D[0])
            T_hms = T.split(":")
            date = Time(f'{D1} {T}') 
            sunday = Time(f'2022-05-01 {T}')
    except IndexError:
        print(D,T)
    return y,m,d,date,sunday,T,T_hms
    

def decret_time_two_month(D,T):
    """
    D - DATE_OBS from table
    T - time of start of observ from function time_format(time_tab) in hh:mm:ss format
    Return separately year, month, day, full date (time function), 
    known sunday (time function), time (hh:mm:ss) and array from time (hh, mm, ss)
    """
       
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[1][2]}-{D[0][1]}-{D[0][0]}".replace(" ","")
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}".replace(" ","")
    T_hms=T.split(":")
    sunday = Time(f'2022-05-01 {T}')

    if len(T_hms[2]) >1:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            date = Time(f"{D1} {T}")
            y = int(D[1][2])
            m = int(D[0][1])
            d = int(D[0][0])
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            date = Time(f"{D2} {T}")
            y = int(D[1][2])
            m = int(D[1][1])
            d = int(D[1][0])
    else:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            date = Time(f"{D1} {T}")
            y = int(D[1][2])
            m = int(D[0][1])
            d = int(D[0][0])
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            date = Time(f"{D2} {T}")
            y = int(D[1][2])
            m = int(D[1][1])
            d = int(D[1][0])
        
    return y,m,d,date,sunday,T,T_hms


def decret_time_two_years(D,T):
    """
    D - DATE_OBS from table
    T - time of start of observ from function time_format(time_tab) in hh:mm:ss format
    Return separately year, month, day, full date (time function), 
    known sunday (time function), time (hh:mm:ss) and array from time (hh, mm, ss)
    """
       
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D1 = f"{D[0][2]}-{D[0][1]}-{D[0][0]}".replace(" ","")
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}".replace(" ","")
    T_hms=T.split(":")
    sunday = Time(f'2022-05-01 {T}')

    if len(T_hms[2]) >1:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            date = Time(f"{D1} {T}")
            y = int(D[0][2])
            m = int(D[0][1])
            d = int(D[0][0])
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            date = Time(f"{D2} {T}")
            y = int(D[1][2])
            m = int(D[1][1])
            d = int(D[1][0])
    else:
        if int(T_hms[0]) >= 10 and int(T_hms[0]) <24: #because obs start not earlier than 16:00 and we have UT, so 16-6=10 
            date = Time(f"{D1} {T}")
            y = int(D[0][2])
            m = int(D[0][1])
            d = int(D[0][0])
        elif int(T_hms[0]) >= 0 and int(T_hms[0]) < 10:
            date = Time(f"{D2} {T}")
            y = int(D[1][2])
            m = int(D[1][1])
            d = int(D[1][0])
        
    return y,m,d,date,sunday,T,T_hms
    
    
def year_check(y,m,d,date,sunday, T, T_hms):

    """
    y - year of observation
    m - month of observation
    d - day of observation
    date - time function of date of observ
    sunday - known sunday 2022-05-01
    T - time of observation in format hh:mm:ss
    T_hms - T splitted by ":"
    """
        
    if y >= 1981 and y < 1991:
        if m == 3:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 7*u.hour
                    else:
                        day_obs = date - 6*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                        continue
                    else:
                        if date < t:
                            day_obs = date - 6*u.hour
                        else:
                            day_obs = date - 7*u.hour
                            
            else:
                day_obs = date - 6*u.hour
                        
        elif m == 9:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 6*u.hour
                    else:
                        day_obs = date - 7*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                        continue
                    else:
                        if date < t:
                            day_obs = date - 7*u.hour
                        else:
                            day_obs = date - 6*u.hour
            else:
                day_obs = date - 7*u.hour
                            
        elif m > 3 and m < 9:
            day_obs = date - 7*u.hour
        else:
            day_obs = date - 6*u.hour
    
    elif y == 1991:
        if m == 3:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 6*u.hour
                    else:
                        day_obs = date - 5*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 5*u.hour
                        else:
                            day_obs = date - 6*u.hour
            else:
                day_obs = date - 5*u.hour
                        
        elif m == 9:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 5*u.hour
                    else:
                        day_obs = date - 6*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 6*u.hour
                        else:
                            day_obs = date - 5*u.hour
            else:
                day_obs = date - 6*u.hour
                
        elif m > 3 and m < 9:
            day_obs = date - 6*u.hour
        else:
            day_obs = date - 5*u.hour
    
    elif y == 1992:
        if m == 1:
            if d <19:
                day_obs = date - 5*u.hour
            else:
                day_obs = date - 6*u.hour
        
        elif m == 3:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 7*u.hour
                    else:
                        day_obs = date - 6*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 6*u.hour
                        else:
                            day_obs = date - 7*u.hour
            else:
                day_obs = date - 6*u.hour
                
        elif m == 9:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 6*u.hour
                    else:
                        day_obs = date - 7*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 7*u.hour
                        else:
                            day_obs = date - 6*u.hour
            else:
                day_obs = date - 7*u.hour
                
        elif m > 3 and m < 9 and m!=1:
            day_obs = date - 7*u.hour
        else: #2,10,11,12
            day_obs = date - 6*u.hour
    
    elif y >= 1993 and y < 1996:
        if m == 3:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 7*u.hour
                    else:
                        day_obs = date - 6*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 6*u.hour
                        else:
                            day_obs = date - 7*u.hour
            else:
                day_obs = date - 6*u.hour
                        
        elif m == 9:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 6*u.hour
                    else:
                        day_obs = date - 7*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 7*u.hour
                        else:
                            day_obs = date - 6*u.hour
            else:
                day_obs = date - 7*u.hour
                
        elif m > 3 and m < 9:
            day_obs = date - 7*u.hour
        else:
            day_obs = date - 6*u.hour
    
    elif y >= 1996 and y < 2005:
        if m == 3:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 7*u.hour
                    else:
                        day_obs = date - 6*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 6*u.hour
                        else:
                            day_obs = date - 7*u.hour
            else:
                day_obs = date - 6*u.hour
                        
        elif m == 10:
            if d >=24 and d <= 31:
                modulo = (date - sunday).value%7
                if modulo == 0:
                    if int(T_hms[0]) >= 3:
                        day_obs = date - 6*u.hour
                    else:
                        day_obs = date - 7*u.hour
                else:
                    t = Time(f"{y}-{m}-23 {T}")
                    modulo = 1
                    while modulo != 0:
                        t = t + 1*u.day
                        modulo = (t - sunday).value % 7
                    else:
                        if date < t:
                            day_obs = date - 7*u.hour
                        else:
                            day_obs = date - 6*u.hour
            else:
                day_obs = date - 7*u.hour
                
        elif m > 3 and m < 10:
            day_obs = date - 7*u.hour
        else:
            day_obs = date - 6*u.hour
    elif y <=1980 or y >= 2005:
        day_obs = date - 6*u.hour
    
    return day_obs


def no_time_obs_one_months(D):
    """
    function for determinate date of obs if there are no time of obs (not for comets)
    """
    D=str(D).split("-")
    D[1] = D[1].split(".")  
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}".replace(" ","")
    date_obs = Time(f'{D2} 00:00:00')
    return date_obs

def no_time_obs_two_months(D):
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}".replace(" ","")
    date_obs = Time(f'{D2} 00:00:00')
    return date_obs

def no_time_obs_two_years(D):   
    D=D.split("-")
    D[0] = D[0].split(".")
    D[1] = D[1].split(".")
    if len(D[1][2]) < 4:#if we have '96' instead of '1996'
        D[1][2] = f"19{D[1][2]}"
    D2 = f"{D[1][2]}-{D[1][1]}-{D[1][0]}".replace(" ","")
    date_obs = Time(f'{D2} 00:00:00')
    return date_obs

def no_time_obs_one_day(D):
    """
    function for determinate date of obs if there are no time of obs (not for comets)
    """
    D = D.split(".")  
    if len(D[2]) < 4:#if we have '96' instead of '1996'
        D[2] = f"19{D[2]}"
    D2 = f"{D[2]}-{D[1]}-{D[0]}".replace(" ","")
    date_obs = Time(f'{D2} 00:00:00')
    return date_obs



def preob_spectra(spectra):
    
    x_log = np.log10(65535/spectra) * 1000
    sum_intens_row = x_log.sum(axis=1)
    return sum_intens_row


def get_band_end1(data):
    data_max = np.max(data) - np.min(data)
    band_ = data_max * 0.3# CHANGED!!!!!!! why0.3
    band = band_ + np.min(data)
    band_end = [band-band_*0.05,band+band_*0.05]

    return band_end

def get_band_end2(data):
    data_max = np.max(data) - np.min(data)
    band_ = data_max * 0.11# CHANGED!!!!!!!
    band = band_ + np.min(data)
    band_end = [band-band_*0.05,band+band_*0.05]

    return band_end

def get_band_end3(data):
    data_max = np.max(data) - np.min(data)
    band_ = data_max * 0.31# CHANGED!!!!!!!
    band = band_ + np.min(data)
    band_end = [band-band_*0.05,band+band_*0.05]

    return band_end

def get_band_end4(data):
    data_max = np.max(data) - np.min(data)
    band_ = data_max * 0.26# CHANGED!!!!!!!
    band = band_ + np.min(data)
    band_end = [band-band_*0.04,band+band_*0.04]

    return band_end

def get_band_end5(data):
    data_max = np.max(data) - np.min(data)
    band_ = data_max * 0.7# CHANGED!!!!!!!
    band = band_ + np.min(data)
    band_end = [band-band_*0.01,band+band_*0.01]

    return band_end

def get_band_end6(data):
    data_max = np.max(data) - np.min(data)
    band_ = data_max * 0.7# CHANGED!!!!!!!
    band = band_ + np.min(data)
    band_end = [band-band_*0.05,band+band_*0.05]

    return band_end

def get_cross_points_ind(sum_intens_row1, bands):

    #среднее скольжение 
    N_rm = 22
    W = np.zeros_like(sum_intens_row1); W[0:N_rm]=1
    FConv = np.fft.fft(sum_intens_row1)*np.fft.fft(W)
    TConv = np.fft.ifft(FConv).real/N_rm
    # отбрасываем те точки, где скользящее среднее неадекватно срабатывает 
    # то есть первые и последние N_rm/2 точек
    # plt.subplots(figsize=(8,5),dpi=100)
    Nskip = int((N_rm-1)/2) 
    X_rm = np.pad(TConv[Nskip:],(0,Nskip),'constant')
    X_rm_int = X_rm.astype('int')

#     plt.scatter(np.arange(0,len(sum_intens_row1))[:-10],X_rm[:-10],color='pink', s=0.5, ls='--', alpha=1, label="medium slip")
#     for i in bands:
#         plt.axhline(i, label = 'BAND')

#     plt.legend()
    #полоса для поиска пересечений по оси Оу
    band_1 = np.arange(int(bands[0]),int(bands[1]),1)        
    #поиск точек пересечения с полосой поиска
    match_1 = np.where(np.isin(X_rm_int ,band_1))[0]    
    #зачистка точек для поиска истинного значения пересечений (с линией, а не полосой)
    cross_points_ind_1 = get_values_with_difference(match_1)  


    return cross_points_ind_1