"""
Group of functions to do a "basic" analysis of SAXS data:
- radial integration: I(q) curve
- power-law extraction: I ~ q^(-beta)
- fitting of scattering peaks to extract their center and width

Notes:
- Everything can be further optimised, taking into account numerical errors (is it necessary)
- Power-law fitting and scattering peak detection needs to be properly tested
- Values optimised for fibrin samples

To add:
- Anisotropy estimation. At the moment several issues with its accuracy/definition.

Created on Fri Nov 24 2017
Last modification on Tue Jan 9 2018
version: 0.0

@author: Cristina MT
"""

class Integrate():
	"""
	Integrates the saxs image, to obtain a 1D curve
	"""

	def radial(saxs_image, mat_qvalues, qres):
		"""
		Defines function that radially integrates the [saxs_image] 2d array
		[mat_qvalues] is a 2D array with the q values,
		it is obtained with the function BuildMatrix.qvalues()
		[qres] is a float number indicating the resolution for the qvector in integration

		OUTPUT:
		[radial_profile] 1D array with the integrated Intensity values
		[q_vector] 1D array with the q values used for integration
		"""

		import numpy as np

		q_vect = np.arange(0, np.max(mat_qvalues.flatten()), step = qres) # q vector for radial integration
		radial_profile = []

		# Radial integration, normalised by the number of pixels taken into account => mean
		for q in range(len(q_vect)-1):
			radial_profile.append(np.mean(saxs_image[(mat_qvalues>q_vect[q])&(mat_qvalues<=q_vect[q+1])].flatten()))

		q_vector = (q_vect[1:]+q_vect[:-1])/2

		return radial_profile, q_vector

class Filtering():

	"""
	Includes all the functions needed to filter data
	"""

	def mirror_logsignal_1D(signal_x, signal_y):
		"""
		Defines function that mirrors a 1D array [signal] in logaritmic scale
		The mirroring is used in kernel filtering by convolution, to avoid edge effects

		Output:
		[offset_nan] integer indicating the index of the NaN values encountered, if any
		[mirror_x][mirror_y] 1D array with the mirrored signal in logarithmic scale.
		It is interpolated to be equally spaced in log scale
		"""

		import numpy as np
		from scipy import interpolate
		import matplotlib.pyplot as plt

		# Identify where the maxima is, to mirror only from that point
		signal_y = np.array(signal_y)
		offset_nan = np.where(signal_y == np.max(signal_y[~np.isnan(signal_y)]))[0][0] + 1

		try: log_signal_y = np.log10(signal_y[offset_nan:])
		except IndexError: log_signal_y = np.log10(signal_y); offset_nan = 1

		log_signal_x = np.log10(signal_x[offset_nan:])

		# Mirror signal, to avoid edge effects in filtering
		mirror_y = np.append(2*log_signal_y[0] - np.flip(log_signal_y[:-10],0), [log_signal_y[:-10], -np.flip(log_signal_y[:-10],0)+2*log_signal_y[-10]])
		mirror_x = np.append(2*log_signal_x[0] - np.flip(log_signal_x[:-10],0), [log_signal_x[:-10], -np.flip(log_signal_x[:-10],0)+2*log_signal_x[-10]])

		# Interpolation to be equally spaced in logscale
		interp_my = interpolate.interp1d(mirror_x, mirror_y, kind = 'slinear')
		mirror_y = interp_my

		return offset_nan, mirror_x, mirror_y

	def gauss_der_1D_fixed(signal, a_scale):
		"""
		Standard wavelet transform using the first derivative of a gaussian as mother wavelet
		The width of the gaussian (proportional to a_scale) filters the image,
		and the function simultaneously derivates the signal

		Note: to preserve the quality of the numerical data,
		[a_scale] should be a float number, multiple of 2.
		"""
		import numpy as np

		fx = np.fft.fftshift(np.fft.fft(signal))
		x = np.arange(-len(signal)/2,len(signal)/2)
		phi = np.exp(-(x/a_scale)*(x/a_scale))  # x*x has better numerical results than x**2
		phi = -(x/a_scale)*phi
		phi_g = phi/np.sum(np.abs(phi))					# Normalised so the filtered signal keeps the same scale of the original
		f_phi = np.fft.fftshift(np.fft.fft(phi_g))      # FFT with freq. shift

		x_filtered = np.abs(np.fft.ifftshift(np.fft.ifft(f_phi*fx)))

		return x_filtered

class Fitting():

	"""
	Class with functions to fit the Radial Profile and extract either the power-law
	dependency of I(q), or to detect the Bragg scattered peak position (if any)
	"""

	def gauss_fit(x, y0, a, x0, sigma):
		"""
		Defines gaussian function to fit the data
		"""
		import numpy as np

		g = y0 + a*np.exp(-(x-x0)**2/(2*sigma**2))

		return g

	def q_powerlaw_lin(qlim, q_vector, radial_profile):
		"""
		Defines function to perform a linear fit on the radial profile data, in logaritmic scale
		Both [q_vector] and [radial_profile] are 1D arrays in linear scale

		Output: [m] scalar value of slope in the linear fit

		Note: Error estimation needs to be added
		"""

		import numpy as np

		ind_0 = np.where(q_vector>=qlim[0])[0][0]
		ind_1 = np.where(q_vector<=qlim[1])[0][-1]

		x = np.log10(q_vector[int(ind_0):int(ind_1)])
		y = np.log10(radial_profile[int(ind_0):int(ind_1)])

		m, b = np.polyfit(x, y, 1)

		return m, b

	def qpeak_loggauss(peak_pos0, qlim, q_vector, corr_radial_profile):

		"""
		Defines function to fit the scattered peak using a gaussian function
		The fit has the purpose of retrieving mainly the peak maxima position,
		since the width estimation is not very accurate

		IMPORTANT: Fit is done on logaritmic scale (seems to be better than linear scale,
		to be confirmed)

		[peak_pos0] scalar value with the expected peak position
		[qlim], 2-element array [qmin,qmax], with the qrange limits for the fit
		[corr_radial_profile] 1D array in LOGARITMIC SCALE of the radial profile
		where the global trendline has been subtracted

		Output:

		[gfit] array containing the fit parameters
		[fit_error] array containing the error estimation of the fit parameters
		[rs_gfit] scalar variable estimating the R-squared for the gaussian fit

		"""

		import numpy as np
		from scipy.optimize import curve_fit

		ind_0 = np.where(q_vector>=qlim[0])[0][0]
		ind_1 = np.where(q_vector<=qlim[1])[0][-1]

		x = np.log10(q_vector[int(ind_0):int(ind_1)])
		y = corr_radial_profile[int(ind_0):int(ind_1)]

		p0 = [-5, 0.1, np.log10(peak_pos0), 0.01]

		gfit, cfit = curve_fit(Fitting.gauss_fit, x, y, p0)
		fit_error = np.sqrt(np.diag(cfit))

		gres = y - Fitting.gauss_fit(x, gfit[0], gfit[1], gfit[2], gfit[3])
		gres_ss = np.sum(gres**2)
		gres_sstot = np.sum(y - np.mean(y)**2)
		rs_gfit = 1 - (gres_ss/gres_sstot)

		return gfit, fit_error, rs_gfit




