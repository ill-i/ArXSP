#from configparser import SafeConfigParser
import io
import os
from unittest import signals
import cv2
import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.time import Time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy import signal ### v1.7.1
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from numpy.polynomial import Polynomial
from typing import Literal
import csv
import re
import random
import string
from datetime import datetime
from scipy.ndimage import rotate as nd_rotate


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
			self.__data = fitsfile[0].data
			self.__header = fitsfile[0].header
			self.__fname = os.path.basename(path).replace(".fits","").replace(".fit","")
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

	def set_data(self,data):
		self.__data = data

	def get_data(self):
		return self.__data

	def get_path(self):
		return self.__path

	def get_fname(self):
		return self.__fname

	def set_fname(self,fname):
		self.__fname = fname

	def get_exptime(self):
		try:
			if 'EXPTIME' in self.__header:
				return self.__header['EXPTIME']
			else:
				return None
		except Exception as e:
			return f'err: {e}'

	def get_dateObs(self):
		try:
			if 'DATE-OBS' in self.__header:
				return self.__header['DATE-OBS']
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
	def get_image(self, colormap):
		if self.__data is None:
			print("err: No data for image!")
			return None
		else:
			norm_image = cv2.normalize(self.__data[::-1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			#return cv2.applyColorMap(norm_image, cv2.COLORMAP_VIRIDIS)
			return cv2.applyColorMap(norm_image, colormap)
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
	def rotate(self, angle=0):
		try:
			# достаём данные и заголовок
			temp_data   = self._arxData.get_data()
			temp_header = self._arxData.get_header()
	
			# безопасно читаем BITPIX (0, если ключа нет)
			bitpix = temp_header.get("BITPIX", 0)
	
			if bitpix > 0:
				# для целочисленных картинок используем OpenCV (warpAffine)
				# dsize = (width, height)
				height, width = temp_data.shape
				shape = (width, height)
				center = (width // 2, height // 2)
	
				matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
				data_rotated = cv2.warpAffine(
					src=temp_data,
					M=matrix,
					dsize=shape
				)
				history_msg = f"Image was rotated on {angle:.3f} deg (OpenCV)"
			else:
				# для float-картинок — scipy.ndimage.rotate без интерполяции
				data_rotated = nd_rotate(
					temp_data,
					angle=angle,
					reshape=False,	# сохраняем исходную форму
					order=0,		  # nearest-neighbor
					mode='nearest'	# граничные пиксели повторяются
				)
				history_msg = f"Image was rotated on {angle:.3f} deg (scipy, order=0)"
	
			# обновляем историю и дату в заголовке
			temp_header.add_history(history_msg)
			temp_header["DATE"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
	
			# возвращаем новый объект ArxData
			return ArxData(None, None, data_rotated, temp_header)
	
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
			temp_header = self._arxData.get_header()
			print("data", data)
			if data is None:
				print("[SDistorsionCorr] Error: нет данных.")
				return None
			
			###====
			data_part = data[top:down]
			print("data_part", data_part)
			#print("data.shape,",data.shape)
			#print("data_part.shape,",data_part.shape())

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
			
			temp_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
			temp_header.add_history(f"Image was processed:s-distortion was corrected, optical density was converted to relative intensity by polinomial from calibrational frames")
			


			newArxDATA = ArxData(None, None, aligned_image1, temp_header)
			return newArxDATA

			#hdu = fits.PrimaryHDU(data=aligned_image1, header=temp_header)

			#return fits.HDUList([hdu])
	
		except Exception as e:
			print(f" Error: {e}")
			return None
	



	def OptDen2Int(self, poly_obj):# best_poly, polynomial_extrapolate):
		try:
			data = self._arxData.get_data()
			temp_header = self._arxData.get_header()
			print(f"data in OptDen2Int={data}")
			best_poly = poly_obj.bestPoly
			print(f"best_poly={best_poly}")
			polynomial_extrapolate = poly_obj.extraPoly
			print(f"polynomial_extrapolate ={polynomial_extrapolate }")
			root = 10#poly_obj.root
			print(f"root ={root}")
			
			if data is None:
				print("[OptDen2Int] Error: no data.")
				return None

			data_image = np.log10(65535 / np.where(data == 0, 1e-32, data)) * 1000
			mask = data_image < root
			#print(f"data in OptDen2Int={data_calib}")
			data_calib = np.empty_like(data_image)
			mask = data_image < root
			data_calib[mask] = polynomial_extrapolate(data_image[mask])
			data_calib[~mask] = best_poly(data_image[~mask])

			temp_header.add_history(f"Optical density was converted to relative intensity")
			temp_header['DATE'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

			print(temp_header)

			return ArxData(None,None,data_calib,temp_header)

		except Exception as e:
			print(f"[OptDen2Int] Error: {e}")
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
			#data = self.get_ArxData_xy()
			
			xy = self.get_ArxData_xy()
			x_vals = xy[:, 0]
			y_vals = xy[:, 1]
			data_arr = np.array(y_vals)
			data_list = data_arr.tolist()
			data = np.array(data_list)
			print("data from xy", data)
			#peak indexes
			peak_x = signal.argrelextrema(data, np.greater, order=order)[0] 
			peak_y =[] 
			for i in range(0,len(data)):
				if i in peak_x:
					# search mean by 5 values: 
					peak_y.append((data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2])/5) 

			#if len(peak_x) !=9 and len(peak_y)!=9:
				#print("Wrong data to find peaks")
			   # return None

			#x and y values of peaks: 
			return xy, peak_x, peak_y

 

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
	   
		xy = self.get_ArxData_xy()
		x_vals = xy[:, 0]
		y_vals = xy[:, 1]
		
		data_smoothed, peak_x = self.__auto_calculate_peaks(y_vals)
		
		values = data_smoothed[peak_x][1:8]
		changes = [y - x for x, y in zip(values, values[1:])]
		increasing = sum(change > 0 for change in changes)
		decreasing = sum(change < 0 for change in changes)
	
		if increasing > decreasing:
			data_smoothed, peak_x = self.__auto_calculate_peaks(data_smoothed[::-1])
			y_vals = y_vals[::-1]
		elif decreasing > increasing:
			pass
		else:
			print("err: Cannot understand calibration peaks values")

		peak_y = data_smoothed[peak_x]
		xy = np.column_stack((x_vals, y_vals))
		return xy, peak_x, peak_y

 
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

	@staticmethod
	def clean_peaks_convert_flux(x_array, y_array):

		c = []
		for i in range(1, len(y_array)):
			c.append(abs(y_array[i] - y_array[i - 1]))
		mean_deviation = np.array(c).sum() / len(y_array)
		
		indexes = []
		for i in range(1, len(y_array)):
			if abs(y_array[i] - y_array[i - 1]) > mean_deviation * 2:
				indexes.append(i)

		y_array = np.delete(y_array, indexes)
		x_array = np.delete(x_array, indexes)
		
		x_flux = 10000 / 10 ** (x_array / 2.5)
		
		paired = sorted(zip(x_flux, y_array), key=lambda p: p[0])
		x_flux_sorted, y_sorted = zip(*paired)
		
		x_flux_sorted = np.array(x_flux_sorted)
		y_sorted = np.array(y_sorted)
		print('x_flux_sorted:',x_flux_sorted)
		print('y_sorted:',y_sorted)

		return x_flux_sorted, y_sorted

	@staticmethod
	def is_smooth(poly, x):
		derivative_coeffs = np.polyder(poly)   # вернет np.poly1d
		derivative_values = derivative_coeffs(x)
		smoothness = np.std(derivative_values)
		return smoothness

	@staticmethod
	def get_poly_mse(x, y, degree):
		coeffs = np.polyfit(x, y, degree)
	   # print('coef')
		poly = np.poly1d(coeffs)
		print("poly",poly)
		y_pred = poly(x)
		mse = ((y - y_pred) ** 2).mean()
		return mse, poly



	@staticmethod
	def is_strictly_monotonic(y_vals):
		diff = np.diff(y_vals)
		print('diff', diff)
		return np.all(diff > 0)

	@staticmethod
	def poly_eval(poly, x):
		"""Безопасная функция для вычисления значений полинома."""
		if poly is None:
			raise ValueError("poly is None")
		return sum(c * x**i for i, c in enumerate(poly.coef[::-1]))
	
	@staticmethod
	def polynomial_extrapolate(coeffs, x):
		"""Экстраполяция по коэффициентам"""
		return sum(c * x**i for i, c in enumerate(coeffs))


	@staticmethod
	def poly_mse_new(x,y):
		MAX_DEGREE = 10
		candidates = [] 
		sorted_indices = np.argsort(x)
		x_sorted = x[sorted_indices]
		y_sorted = y[sorted_indices]
		
		print("check x_sorted",x_sorted)
		print("check y_sorted",y_sorted)

		print("check poly_mse_new")

		# Вспомогательные данные
		x_test = np.linspace(min(x_sorted), max(x_sorted), 500)

		for order_mse in range(1, MAX_DEGREE + 1):
			try:
				mse, poly = ArxCollibEditor.get_poly_mse(x_sorted, y_sorted, order_mse)
				y_pred = poly(x_test)
	
				if ArxCollibEditor.is_strictly_monotonic(y_pred):
					smoothness = ArxCollibEditor.is_smooth(poly, x_test)
					candidates.append((order_mse, poly, smoothness))
	
			except Exception as e:
				print(f"Ошибка на степени {order_mse}: {e}")
			
		x_test = np.linspace(min(x_sorted), max(x_sorted), 500)

		if candidates:
			sorted_candidates = sorted(candidates, key=lambda x: x[2])
			median_candidate = sorted_candidates[len(sorted_candidates) // 2]
			best_poly = median_candidate[1]
			#plt.savefig(filepath)
			print(f"Best degree: {median_candidate[0]}, MSE: {mse}")
		else:
			best_poly = None
			print("best_poly = None.")

		# Вычисление корня
		root = fsolve(lambda x: ArxCollibEditor.poly_eval(best_poly, x), 1)[0]

		# Построение полинома экстраполяции
		x_points = [0, x_test[0]]
		y_points = [0, best_poly(x_test)[0]]
		extrapolation_poly = Polynomial.fit(x_points, y_points, 20).convert()
		extrapolation_coefficients = extrapolation_poly.coef

		# Возвращаем всё нужное
		return best_poly, extrapolation_coefficients, root




#End of ArxCollibEditor

class PeaksItem(object):

#Initializer:
	def __init__(self, peaks, date,expTime):
		self.__date = date
		self.__expTime =expTime
		self.__peaks = peaks

	def get_peaks(self):
		return self.__peaks

	def get_date(self):
		return self.__date

	def get_expTime(self):
		return self.__expTime
	 
###########################################




class CsvPolynom(object):
	
#Initializer:
	def __init__(self, date, bestPoly, extraPoly, root, Id = None, note = None):

		self.Id = Id 
		self.note = note
		self.root = root

		formatted_date = CsvPolynom.isValidDate(date)
		if formatted_date:
			self.date = formatted_date
		else:
			raise ValueError("Wrong Date format! Expected FITS date string like 'YYYY-MM-DDThh:mm:ss.sss'")

		if not isinstance(bestPoly, np.poly1d):
			raise TypeError("bestPoly should be an object of numpy.poly1d")
		if not isinstance(extraPoly, np.poly1d):
			raise TypeError("extraPoly should be an object of numpy.poly1d")
		
		if bestPoly.order < 0 or np.allclose(bestPoly.coeffs, 0):
			raise ValueError("bestPoly should not be an empty")
		if extraPoly.order < 0 or np.allclose(extraPoly.coeffs, 0):
			raise ValueError("extraPoly should not be an empty")

		self.bestPoly = bestPoly
		self.extraPoly = extraPoly
#...........................................................................


#To String:
	def __str__(self):
		return (f"CsvPolynom(date={self.date}, "
				f"bestPoly={self.bestPoly}, "
				f"extraPoly={self.extraPoly}, "
				f"root='{self.root}', "
				f"Id={self.Id}, "
				f"note='{self.note}')")



#Validation date format:
	@staticmethod
	def isValidDate(date_str):
		try:
			dt = datetime.strptime(date_str, "%d.%m.%Y")
			return date_str  # Уже в нужном формате — возвращаем как есть
		except ValueError:
			pass

		try:
			t = Time(date_str, format='fits')
			dt = t.to_datetime()
			return dt.strftime("%d.%m.%Y")  # Преобразуем в нужный формат
		except Exception:
			return False
#..................................................

#Generate id to write on file:
	def generate_id(self, csv_path=None):

		existing_ids = set()
		csv_path = csv_path or "calibration_polynomial.csv"
		
		if os.path.isfile(csv_path):
			with open(csv_path, mode='r', newline='') as file:
				reader = csv.DictReader(file)
				existing_ids = {row["id"] for row in reader if "id" in row}
				
		while True:
			new_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
			if new_id not in existing_ids:
				return new_id
#.........................................................................................

#Save polynom in the file
	def savePoly(self, csv_path=None):
		csv_path = csv_path or "calibration_polynomial.csv"

		self.Id = self.generate_id(csv_path)

		row = [
			self.Id,
			self.date,
			";".join(map(str, self.bestPoly.coeffs)),
			";".join(map(str, self.extraPoly.coeffs)),
			str(self.root),
			self.note
		]

		file_exists = os.path.isfile(csv_path)
		with open(csv_path, mode='a', newline='') as file:
			writer = csv.writer(file)
			if not file_exists:
				writer.writerow(["id", "date_obs", "best_poly", "extra_poly", "root", "note"])
			writer.writerow(row)
			print("File with poly saved!")
#End of Save..................................................................................
	

#Upload the csv polynom:
	@staticmethod
	def upLoadCsv(path):
		path = path or "calibration_polynomial.csv"
		polynoms = []
		if not os.path.isfile(path):
			print(f"File {path} not found.")
			return polynoms

		with open(path, mode='r', newline='') as file:
			reader = csv.DictReader(file)
			for row in reader:
				try:
					best_coeffs = list(map(float, row["best_poly"].split(";")))
					extra_coeffs = list(map(float, row["extra_poly"].split(";")))
					best_poly = np.poly1d(best_coeffs)
					extra_poly = np.poly1d(extra_coeffs)

					poly = CsvPolynom(
						date=row["date_obs"],
						bestPoly=best_poly,
						extraPoly=extra_poly,
						root=float(row["root"]) if row["root"] else None,
						Id=row["id"],
						note=row.get("note")
					)
					polynoms.append(poly)
				except Exception as e:
					print(f"Error during reading row: {row}\n{e}")
		return polynoms

	@staticmethod
	def findNearestDate(poly_list, target_date_str):
		def parse_date(s):
			s = s.strip()
			if 'T' in s:
				s = s.split('T')[0]  # отбрасываем время, оставляем только дату
			for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y"):
				try:
					return datetime.strptime(s, fmt)
				except ValueError:
					continue
			return None

		def get_date_range(date_str):
			d = parse_date(date_str)
			if d:
				return (d, d)

			parts = re.split(r'\s*[-–]\s*', date_str.strip())

			if len(parts) == 2:
				try:
					p1, p2 = parts[0].strip(), parts[1].strip()
					if '.' not in p1 and '.' in p2:  # 25-27.05.2023
						day1 = int(p1)
						day2, month, year = map(int, p2.split('.'))
						d1 = datetime(year, month, day1)
						d2 = datetime(year, month, day2)
					elif p1.count('.') == 1 and '.' in p2:  # 25.04-27.05.2023
						year = int(p2.split('.')[-1])
						d1 = parse_date(p1 + f".{year}")
						d2 = parse_date(p2)
					else:
						d1 = parse_date(p1)
						d2 = parse_date(p2)
					return (d1, d2) if d1 and d2 else None
				except Exception:
					return None

			return None

		date_range = get_date_range(target_date_str)
		if not date_range:
			raise ValueError("Wrong date format. Allowed: dd.mm.yyyy, dd-dd.mm.yyyy, dd.mm-dd.mm.yyyy, dd.mm.yyyy-dd.mm.yyyy, yyyy-mm-dd, yyyy-mm-dd - yyyy-mm-dd, yyyy-mm-ddTHH:MM:SS")

		start_date, end_date = date_range

		def date_distance(poly):
			poly_date = parse_date(poly.date)
			if not poly_date:
				return float('inf')
			if start_date <= poly_date <= end_date:
				return 0
			return min(abs((poly_date - start_date).days), abs((poly_date - end_date).days))

		return min(poly_list, key=date_distance) if poly_list else None
#End of Upload csv...........................................................


#End of Csv Polynom___________________________________________________________________________


