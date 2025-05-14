import io
import os
from unittest import signals
import cv2
import numpy as np
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy import signal ### v1.7.1
from scipy.signal import find_peaks
from typing import Literal
from numpy.polynomial import Polynomial     
from scipy.optimize import fsolve 

class ArxData(object):
    
#Initialize:
    def __init__(self, path=None, fname = None, data = None, header = None):

    #Main data created by code:
        if path is None:
            self.__data = data
            self.__header = header
            self.__fname = fname
            return

    #Main data created by file:
        self.__path = path
        self.__data = None
        self.__header = None
        self.__fname = fname

    #Read main data:
        try:
            fitsfile = fits.open(path)
            print(f"[ArxData] FITS opened: {path}")
            print(f"[ArxData] HDU count: {len(fitsfile)}")
            print(f"[ArxData] HDU types: {[type(hdu) for hdu in fitsfile]}")
            print(f"[ArxData] Primary HDU type: {type(fitsfile[0].data)}")

            self.__data = fitsfile[0].data
            self.__header = fitsfile[0].header
            self.__fname = os.path.basename(path).replace(".fit","").replace(".fits","")
            print(self.__fname)

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            return
#End of init_________________________________________

#Getters:
    def get_header(self):
        return self.__header

    def set_header(self, header):
        self.__header = header

    def get_data(self):
        return self.__data

    def get_path(self):
        return self.__path

    def get_fname(self):
        return self.__fname

    def get_exptime(self):
        try:
            if 'EXPTIME' in self.__header:
                return self.__header['EXPTIME']
            else:
                return None
        except Exception as e:
            return f'err: {e}'


#End of getters_____________

#Save as png in buffer:
    def get_png_buff(self):

        if self.__data is None:
            print("err: No data for .png!")
            return None
        else:
            buff = io.BytesIO() 
            plt.imsave(buff, self.__data[::-1], format='png') 
            # Перемещаем курсор в начало
            buff.seek(0)  
            return buff 
#End of Save as png in buffer________________________________


#Get image as cv2-normalized:
    def get_image(self):
        if self.__data is None:
            print("err: No data for image!")
            return None
        else:
            norm_image = cv2.normalize(self.__data[::-1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.applyColorMap(norm_image, cv2.COLORMAP_VIRIDIS)
#End of Get image as cv2-normalized______________________________________________________________________


#Save as png in path:
    def save_as_png(self, path = None):
        if self.__data is None:
            print("err: No data for .png!")
            return
        
        fname = f"{self.__fname}.png"
        
        if path is None:
            plt.imsave(fname, self.__data[::-1], format='png')  
        else:
            full_path = os.path.join(path, fname)
            plt.imsave(full_path, self.__data[::-1], format='png')
#End of save_as_png_______________________________________________

#Save as fits:
    def save_as_fits(self, path=None):

        if self.__data is None or self.__header is None:
            print("err: No data or header for .fits!")
            return
        fits_file = fits.PrimaryHDU(data=self.__data, header=self.__header)
        fits_list = fits.HDUList([fits_file])
        if path is None:
            fits_list.writeto(self.__fname+'.fits',overwrite=True) 
        else:
            full_path = os.path.join(path, self.__fname)
            fits_list.writeto(full_path+'.fits',overwrite=True) 
#End of Save as png in path_______________________________________
#End of class ArxData_____________________________________________


from datetime import datetime
#class ArxDataEditor(ABC):
class ArxDataEditor(object):

#initialize:
    def __init__(self, arxData):

        if arxData is None:
            print("argument err. arxData is None")
            return

        if not isinstance(arxData, ArxData):
            print("arxData type error")
            return

        self._arxData = arxData
#End of initializer...............................


#Rotation by angle:
    def rotate(self,angle=0): 
        try: 
            temp_data = self._arxData.get_data()
            temp_header =self._arxData.get_header()

            shape = (temp_data.shape[1],temp_data .shape[0])
            center = (int(shape[0]/2),int(shape[1]/2))
            matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1 ) 	
        
            data_rotated = cv2.warpAffine(src=temp_data, M=matrix, dsize=shape) 
        
            temp_header.add_history(f"Image was rotated on {angle:.1f} deg")
            temp_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

            return ArxData(None,None,data_rotated,temp_header)
        except Exception as e:
            print(f"[rotate] Error occurred: {e}")
            return None

        
#End of Rotation.............

#Cropping:
    def crop(self, right, left, top, down):
        try: 
            temp_data = self._arxData.get_data()
            temp_header = self._arxData.get_header()
            
            if temp_data is None or not isinstance(temp_data, np.ndarray):
                raise TypeError("Input data must be a valid 2D NumPy array.")
            if temp_data.ndim != 2:
                raise ValueError("Only 2D arrays are supported for cropping.")
            
            w = temp_data.shape[1]
            h = temp_data.shape[0]

            Left = int(left * w / 100)
            Right = int(right * w / 100)
            Top = int(top * h / 100)
            Down = int(down * h / 100)

            # Проверка на допустимость обрезки
            if Left + Right >= w:
                raise ValueError(f"Invalid crop: left + right ({Left + Right}px) >= image width ({w}px).")
            if Top + Down >= h:
                raise ValueError(f"Invalid crop: top + down ({Top + Down}px) >= image height ({h}px).")

            data_croped = temp_data[Down:(h - Top), Left:(w - Right)]
            
            temp_header["NAXIS1"] = abs(Left - (w - Right))
            temp_header["NAXIS2"] = abs(Down - (h - Top))
            temp_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
            temp_header.add_history(f"Image was cropped: L={left}%, R={right}%, T={top}%, D={down}%")
            
            return ArxData(None, None, data_croped, temp_header)
        except Exception as e:
            print(f"[crop] Error occurred: {e}")
            return None

#End of Cropping.......................................................................................


#Obtain Optical Density data:
    def get_OpticalDens(self):
        data = np.where((self._arxData.get_data()==0),1e-32,self._arxData.get_data())
        return np.log10(65535/data)*1000
#End of Optical data.......................................


#Get Arx_Data.data as array:
    def get_ArxData_xy(self):
        optic_data = self.get_OpticalDens()

        array_y = np.mean(optic_data, axis=0)  
        return np.column_stack((np.arange(array_y.shape[0]), array_y))

#End of get_ArxData_xy________________________

    #def get_max_exposition(self, )

#End of ArxDataEditor



class ArxSpectEditor(ArxDataEditor):

## add other func
    def SDistorsionCorr(self, top=None, down=None, order_mse=None):
        try:
            # Получение данных
            data = self._arxData.get_data()
            if data is None:
                print("[SDistorsionCorr] Error: нет данных.")
                return None
            
            data_part = data[top:down]
            data_columns = []
            for i in range(0,len(data_part[0])):
                    data_columns.append(list(data_part[:,i]))
            
            data_columns=list(data_columns)
            
            data_collumns_entire_image = []
            for j in range(0,len(data[0])):
                data_collumns_entire_image.append(list(data[:,j]))
            
            data_collumns_entire_image = list(data_collumns_entire_image)
            
            data_col_min = []
            for i in range(0,len(data_columns)):
                    data_col_min.append(np.min(data_columns[i]))
            
            index1 = []
            for i in range(0,len(data_col_min)):
                index1.append(data_columns[i].index(data_col_min[i]))
            
            median1 = []
            n = 0
            while n+300<len(index1):
                k=n+300
                median1.append([n,k,np.median(index1[n:k])])
                n=k
            else:
                median1.append([n,-1,np.median(index1[n:])])

            delt1=[]
            for i in range(1,len(index1)):
                delt1.append(abs(index1[i]-index1[i-1]))
            delt_mean1 = np.array(delt1).sum()/len(delt1)    
    
            for i in median1:
                for j in range(i[0],i[1]):
                    if abs(i[2]-index1[j])>delt_mean1:
                        index1[j] = i[2]
            
            xp1 = np.arange(0,len(index1))

            order_mse = order_mse
            len_x = len(xp1)
            previous_mse = None  # Initialize as None

            while True:
                z1, residuals, *_ = np.polyfit(xp1, index1, order_mse, full=True)
                mse = residuals[0] / len_x
                if len(residuals) == 0:
                    break
                if previous_mse is not None and mse != 0:
                    change_percentage = abs((mse - previous_mse) / previous_mse)
                    if change_percentage <= 0.01:  # Check if change is <= 1%
                        order_mse = order_mse - 1
                        break
                
                order_mse += 1
                previous_mse = mse

            polynomial1 = np.poly1d(z1)
            y_polynomial1 = polynomial1(xp1)

            xp_tg1 = np.arange(0,len(y_polynomial1))
            z_tg1 = np.polyfit(xp_tg1,y_polynomial1, 1)
            polynomial_tg1 = np.poly1d(z_tg1)
            y_polynomial_tg1 = polynomial_tg1(xp_tg1)
            tg1 = polynomial_tg1[1]
            mean_polynomial1 = y_polynomial1.mean()

            delta_y1 = []
            for i in range(0,len(index1)):
                delta1 = mean_polynomial1 - polynomial1(xp1)[i]
                delta_y1.append(int(round(delta1)))
            
            shape1 = np.array(data_collumns_entire_image).shape
            new_pic_col1 = np.zeros(shape1,dtype="uint16")

            for i in range(0,len(data_collumns_entire_image)):
                for j in range(0,len(data_collumns_entire_image[i])):
                    m  = j + delta_y1[i]
                    if m<len(data_collumns_entire_image[i]):
                        new_pic_col1[i][m] = data_collumns_entire_image[i][j] 
                    else:
                        m = m - len(data_collumns_entire_image[i])
                        new_pic_col1[i][m] = data_collumns_entire_image[i][j]  
    
            data_rows1 = []
            for i in range(0,len(new_pic_col1[0])):
                data_rows1.append(list(new_pic_col1[:,i]))
    
            delta_x1 = np.arange(0,len(data_rows1))*tg1
            shape1 = np.array(data_rows1).shape
            new_data_rows1 = np.zeros(shape1,dtype="uint16")
            for i in range(0,len(data_rows1)):
                for j in range(0,len(data_rows1[i])):
                    m  = int(round(j + 2*delta_x1[i]))
                    if m<len(data_rows1[i]):
                        new_data_rows1[i][m] = data_rows1[i][j] 
                    else:
                        m = m - len(data_rows1[i])
                        new_data_rows1[i][m] = data_rows1[i][j] 
    
            aligned_image1 = new_data_rows1       
            return aligned_image1
    
        except Exception as e:
            print(f" Ошибка: {e}")
            return None
    



#End of ArxSpectEditor




class ArxCollibEditor(ArxDataEditor):

#initialize:
    def __init__(self, arxData):
    # parent
        super().__init__(arxData)
#End of initializer...............................



#Find peaks:
    def get_peaks(self, order = None):
    #order mode:
        if order is None:
            return self.get_peaks_auto()
        else: 
            if order > 21: order = 21
            #get mean data of optical dens:
            data = self.get_ArxData_xy()
            #peak indexes
            peak_x = signal.argrelextrema(data, np.greater, order=order)[0] 
            peak_y =[] 
            for i in range(0,len(data)):
                if i in peak_x:
                    # search mean by 5 values: 
                    peak_y.append((data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2])/5) 

            if len(peak_x) !=9 and len(peak_y)!=9:
                print("Wrong data to find peaks")
                return None

            #x and y values of peaks: 
            return peak_x, peak_y
#End of Find peaks____________________________________________________________________

    @staticmethod
    def shift_peaks_magnitude(arg: Literal["before", "after"], shift_value, max_exposition):
                # attenuators of magnitude:
        attenuator = [0,0.61,1.1,1.47,1.84,2.25,2.66,3.04,0][:-1]

        if arg == "before":
            attenuator = [0,0.61,1.1,1.47,1.84,2.25,2.66,3.04,0][:-1]
        else: 
            #after:
            attenuator = [0,0.5,0.97,1.44,1.93,2.43,2.69,3.04,0][:-1]

        sign = 0

        if shift_value < 0: 
            sign = -1 
        else:
            sign = 1

        if shift_value < -100 or shift_value > 100:
            delta = 0
        else: delta = 2.5 * np.log(max_exposition/((max_exposition*max(abs(shift_value), 1))/100)) 

        return attenuator + delta*sign

    









#Find peaks automatically:
    def get_peaks_auto(self):
        
        data = self.get_ArxData_xy()

        data_smoothed, peak_x = self.__auto_calculate_peaks(data)
        values = data_smoothed[peak_x][1:8]
        changes = [y - x for x, y in zip(values, values[1:])]
        increasing = sum(change > 0 for change in changes)
        decreasing = sum(change < 0 for change in changes)
    
        if increasing > decreasing:
            data_smoothed, peak_x = self.__auto_calculate_peaks(data[::-1])
        elif decreasing > increasing:
            pass
        else:
            print("err: Cannot understand calibration peaks values")

        peak_y = data_smoothed[peak_x]
        return peak_x, peak_y
#End of Find peaks automatically_________________________________________



#Private method to auto calculate peaks:
    def __auto_calculate_peaks(self,data):
        # Step 1: Define the window size for the moving average.
        N_rm = 5  # Size of the moving average window
    
        # Step 2: Create a window for moving average and perform convolution in the frequency domain.
        W = np.zeros_like(data); W[:N_rm] = 1  # Moving average window
        FConv = np.fft.fft(data) * np.fft.fft(W)  # Convolution in frequency domain
    
        # Step 3: Convert back to time domain and normalize to obtain the smoothed data.
        TConv = np.fft.ifft(FConv).real / N_rm  # Inverse FFT for smoothing
        data_smoothed = np.pad(TConv[(N_rm-1)//2:], (0, (N_rm-1)//2), 'constant')  # Padding for consistency
    
        # Step 4: Detect peaks with a minimum distance constraint to avoid closely-spaced false positives.
        all_peaks, _ = find_peaks(data_smoothed, distance=len(data_smoothed)/20)
    
        # Step 5: Ensure there are enough peaks to analyze.
        if len(all_peaks) < 2:
            return [], []  # Early exit if not enough peaks found
    
        # Step 6: Divide the data into two halves and find the highest peak in each half.
        mid_point = len(data_smoothed) // 2
        first_half_peaks = all_peaks[all_peaks < mid_point]
        second_half_peaks = all_peaks[all_peaks >= mid_point]
        if len(first_half_peaks) == 0 or len(second_half_peaks) == 0:
            return [], []  # Early exit if one half lacks peaks

        # Identifying the highest peaks in each half.
        first_peak = first_half_peaks[np.argmax(data_smoothed[first_half_peaks])]
        last_peak = second_half_peaks[np.argmax(data_smoothed[second_half_peaks])]

        # Step 7: Select additional peaks between the highest peaks of each half, limited to the seven highest.
        internal_peaks = all_peaks[(all_peaks > first_peak) & (all_peaks < last_peak)]
        if len(internal_peaks) > 7:
            internal_peaks = internal_peaks[np.argsort(data_smoothed[internal_peaks])[-7:]]  # Focus on significant features
        elif len(internal_peaks) < 7:
            data_smoothed = data
            add_shift = 20 #чтобы не захватывал крайние пики, которые итак учтены
            internal_peaks, _ = find_peaks(data_smoothed[first_peak+add_shift:last_peak-add_shift], distance=len(data_smoothed[first_peak:last_peak])/100)
            internal_peaks = internal_peaks + first_peak+add_shift
            if len(internal_peaks) > 7:
                print("len(internal_peaks) > 7")
                internal_peaks = internal_peaks[np.argsort(data_smoothed[internal_peaks])[-7:]]  # Focus on significant features
       
        # Combine, sort, and return the indices of selected peaks.
        peak_indx = np.sort(np.concatenate(([first_peak], internal_peaks, [last_peak])))
        if len(peak_indx)!=9:
            return "Couldn't find all peaks. Please write an order and try again"
        return data_smoothed, peak_indx
#End of Privates________________________________________________________________________________________________________





    def fit_monotonic_poly(self, x, y):

        MAX_DEGREE = 15
        candidates = []
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        for degree in range(1, MAX_DEGREE + 1):
            mse, poly = self.get_poly_mse(x_sorted, y_sorted, degree)
            y_pred = poly(x_sorted)
            if self.is_strictly_monotonic(y_pred):
                smoothness = self.is_smooth(poly, x_sorted)
                candidates.append((degree, poly, smoothness))

        if not candidates:
            raise ValueError("No suitable monotonous polynomial was found.")

        sorted_candidates = sorted(candidates, key=lambda c: c[2])
        median_candidate = sorted_candidates[len(sorted_candidates) // 2]
        best_poly = median_candidate[1]

        def poly_function(xv):
            return best_poly(xv)

        root = fsolve(poly_function, 1)[0]

        # Extrapolation
        x_test = np.linspace(min(x_sorted), max(x_sorted), 500)
        x_points = [0, x_test[0]]
        y_points = [0, best_poly(x_test[0])]
        extrapolation_poly = Polynomial.fit(x_points, y_points, 20).convert()

        def polynomial_extrapolate(xv):
            return extrapolation_poly(xv)

        return best_poly, polynomial_extrapolate, root
        
    @staticmethod
    def get_poly_mse(x, y, degree):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        mse = ((y - y_pred) ** 2).mean()
        return mse, poly

    @staticmethod
    def is_strictly_monotonic(y_vals):
        diff = np.diff(y_vals)
        return np.all(diff > 0) or np.all(diff < 0)

    @staticmethod
    def is_smooth(poly, x):
        derivative = np.polyder(poly)
        derivative_values = derivative(x)
        return np.std(derivative_values)

    def plot_polynomials(self,  x, y, best_poly, polynomial_extrapolate):

        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        x_test = np.linspace(min(x_sorted), max(x_sorted), 500)
        x_extrapolate = np.linspace(0, x_test[0], 100)
        y_extrapolate = [polynomial_extrapolate(xv) for xv in x_extrapolate]

        x_full = np.concatenate([x_extrapolate, x_test])
        y_full = np.concatenate([y_extrapolate, best_poly(x_test)])

        plt.figure(figsize=(10, 6))
        plt.scatter(x_sorted, y_sorted, color='blue', s=20, label='Actual Data')
        plt.plot(x_full, y_full, 'r--', label='Fitted + Extrapolated Polynomial')
        plt.legend()
        plt.show()

        
#End of ArxCollibEditor
####################################################################################################################













#Add Later:
####################################################################################################################
#data = self.__arxData.get_data()

#        shape = (data.shape[1],data.shape[0])
#        #print(shape)
#        center = (int(shape[0]/2),int(shape[1]/2))
#        matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1 )
#        data_rotated = cv2.warpAffine(src=data, M=matrix, dsize=shape) 
        
#        return ArxData(None,f"rot_{angle}_{self.__arxData.get_fname()}", data_rotated, self.__arxData.get_header())
####################################################################################################################