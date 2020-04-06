"""
Test code for analysing SAXS data from ESRF

Created on Fri Nov 24 2017
Last modification on Mon Jan 8 2018
version: 0.0

@author: Cristina MT
"""

# Import modules and functions
import numpy as np; import matplotlib.pyplot as plt
from scipy import interpolate

from saxs_open_v0 import OpenSAXS, ShowSAXS
from saxs_matrix_v0 import BuildMatrix
from saxs_basic_v0 import Integrate, Filtering, Fitting

# ========================================================================================
# Files to read
filedir = '[Replace with base file directory]'

file_poni = filedir + '[Replace with .poni file]'
file_mask = filedir + '[Replace with mask file]'
file_blank = filedir + '[Replace with .edf file containing the blank]'
file_sample = filedir + '[Replace with .edf file containing the sample]'

# Variables to choose how data is processed
norm_factor = True    # Additional normalisation of background image.
					  # Set to True to extract the value from calibration information in poni file
					  # Set to False to ignore it (value equals 1)
					  # Set to a value between 0-1 to adjust manually the value
					  # Note: It's not clear what is this normalisation, to be checked with Daniel
res_vect = 1    # Resolution of the radial integration, in pixels. Recommended value: 1
a_der = 5.  # Width of the gaussian filter used to subtract scattering trendline in radial profile
qmin_pl = 0.05  # Lower limit to fit log(I) = beta*log(q)
qmax_pl = 0.15	# Upper limit
peak_pos0 = 0.3 # Expected position of scattering peak, used for gauss fit
qmin_fg = 0.22	# Lower limit to fit the scattered peak
qmax_fg = 0.38	# Upper limit

# Display 'checkpoints'
checkpoint_0 = 0  # Raw data check
checkpoint_1 = 0  # Background subtraction in 2D image
checkpoint_2 = 0  # Radial profile

# ========================================================================================
# ========================================================================================

# Read files and store content in variables
cal_info = OpenSAXS.read_poni(file_poni)

if file_mask[-3:] == 'edf' : mask = OpenSAXS.read_mask_edf(file_mask)
elif file_mask[-3:] == 'txt' : mask = OpenSAXS.read_mask_txt(file_mask)

img_blank, header_blank = OpenSAXS.read_edf(file_blank)
img_sample, header_sample = OpenSAXS.read_edf(file_sample)


# --------------------------------------------
# CHECKPOINT 0 : Correct reading of raw data.
# Display the input data

if checkpoint_0 == 1:
	ShowSAXS.mask(mask)	# Shows mask (binary)
	ShowSAXS.raw_img(img_blank, vmin = 3, vmax = 8)	# Shows blank image
	plt.title('...'+file_blank[-30:])
	ShowSAXS.raw_img(img_sample, vmin = 3, vmax = 8)	# Shows sample image
	plt.title('...'+file_sample[-30:])
	ShowSAXS.img_wmask(img_sample, mask, vmin = 3, vmax = 8) # Shows overlay mask on sample image
	plt.show()

# ========================================================================================

# Variables used for normalisation, obtained from calibration info in poni file

photo_sample = np.float(header_sample.get('Photo'))
monitor_sample = np.float(header_sample.get('Monitor'))

photo_blank = np.float(header_blank.get('Photo'))
monitor_blank = np.float(header_blank.get('Monitor'))

# Choose the given normalisation factor for background (See notes in line ~24 above
if norm_factor == True: abs_corr = (photo_sample/monitor_sample)/(photo_blank/monitor_blank)
elif norm_factor == False: abs_corr = 1.
else: abs_corr = 1.*norm_factor

# Normalise each saxs data and subtract background from sample
img_corr = (img_sample/photo_sample) - (img_blank/photo_blank)*abs_corr
img_corr[mask==1] = -1

# ------------------------------------------------------------
# CHECKPOINT 1: Data normalisation and background subtraction

if checkpoint_1 == 1:
	ShowSAXS.raw_img(img_corr, vmin = -9, vmax = -3)
	plt.title('Corrected Image sample')
	plt.show()

# ========================================================================================

# Build coordinate matrices used for analysis
center, mat_dist = BuildMatrix.distance(img_sample.shape, cal_info) # Distance matrix (in m)
mat_qvalues = BuildMatrix.qvalues(mat_dist, cal_info)  # matrix of q values
mat_qvalues[mask == 1] = -1

# Radial integration, the resolution of the qvector [res_vect] is given in pixels,
# needs to be converted o q value
qdiff = np.abs(mat_qvalues[int(center[0]):,1:] - mat_qvalues[int(center[0]):,:-1])
qmin = np.mean(qdiff[qdiff>0])
qres = res_vect*qmin

# Plots I(q) in loglog scale

radial_profile, q_vector = Integrate.radial(img_corr, mat_qvalues, qres)
radial_profile = np.array(radial_profile)
q_vector = np.array(q_vector)

# ------------------------------------------------------------
# CHECKPOINT 2: Radial profile

if checkpoint_2 == 1:

	plt.figure()
	plt.loglog(q_vector, radial_profile, c = 'k')
	plt.xlabel('q (1/m)')
	plt.ylabel ('I(q) (a.u.)')
	plt.show()

# ========================================================================================

# Fit line to log(I) = beta*log(q) to extract power-law, in the range qmin_pl, qmax_pl
beta = Fitting.q_powerlaw_lin([qmin_pl, qmax_pl], q_vector, radial_profile)

# Mirror the radial profile, and then apply first derivative of a gaussian as a kernel
# If the kernel is applied with a width large enough, it should subtract the trendline
# Less subjective way of subtracting the 'regular scattering' profile from the scattering peak

# All the interpolated mirrored signals, are in logscale
offset_nan, mirror_qv, mirror_rp = Filtering.mirror_logsignal_1D(q_vector, radial_profile)
interp_mq = np.linspace(mirror_qv[0], mirror_qv[-1], np.power(int(len(mirror_qv)/2),2))  # interpolation, power of 2 for filtering
interp_mr = mirror_rp(interp_mq)

# Choose filter width based on fraction of the one order of magnitude
number_decades = len(np.unique([int(mq) for mq in interp_mq]))
a_scale_der = int(len(interp_mr)/number_decades/a_der)

# Derivate and extract local trendline from radial profile
der_rp = Filtering.gauss_der_1D_fixed(interp_mr, float(a_scale_der))
trendline_rp = -der_rp*(interp_mq)-interp_mq
corr_radial_profile = interp_mr-trendline_rp

# Go back to measured q range, still in logspace
inv_interp_corr_rp = interpolate.interp1d(
								interp_mq, corr_radial_profile, kind = 'slinear')
inv_corr_radial_profile = inv_interp_corr_rp(np.log10(q_vector[offset_nan:]))

# Fit scattering peak in logspace  (Note: Is it better?)
peak_gfit, error_gfit, rs_gfit = Fitting.qpeak_loggauss(
							peak_pos0, [qmin_fg, qmax_fg],
							q_vector[offset_nan:],
							inv_corr_radial_profile)
peak_pos = np.power(10, peak_gfit[2])
peak_width = np.power(10, 2*np.sqrt(2*np.log(2))*peak_gfit[3])
width_error = 2*np.sqrt(2*np.log(2))*error_gfit[3]
lq = 2*np.pi/peak_pos  # Characteristic length of q value

# ========================================================================================
# ========================================================================================

print('Peak position: ' + str(peak_pos) +\
		'\nPeak position error: ' +  str(error_gfit[2]) +\
		'\nPeak width: ' + str(peak_width) +\
		'\nPeak width error: ' +str(width_error) +\
		'\nR-square Fit: ' +str(rs_gfit) +\
		'\nCharacteristic length: ' +str(lq)
		)

plt.figure()
plt.loglog(q_vector, radial_profile, c = 'k', label = 'Data')
plt.loglog(q_vector[offset_nan:],
			np.power(10, inv_corr_radial_profile),
			c = [0.7,0.7,0.7],
			label = 'Trendline correction')
plt.loglog([qmin_pl,qmax_pl],
			np.min(radial_profile[~np.isnan(radial_profile)])*([qmin_pl, qmax_pl]**beta),
			c = 'c',
			label = 'Power-law fit')
plt.text(qmax_pl,
		2*np.min(radial_profile[~np.isnan(radial_profile)])*qmax_pl**beta,
		r'$\beta$ = ' +str(-beta)[:4])
plt.loglog(q_vector[offset_nan:],
			np.power(10,Fitting.gauss_fit(
					np.log10(q_vector[offset_nan:]),
					peak_gfit[0], peak_gfit[1], peak_gfit[2], peak_gfit[3])),
			'-', c = 'cyan', lw = 1,
			label = 'Gauss Fit Peak')
plt.loglog([peak_pos,peak_pos], [0,1], '--', lw = 1, c = 'm')
plt.text(peak_pos*1.5, np.mean(radial_profile[~np.isnan(radial_profile)]),
		'q = ' + str(peak_pos)[:5] +r'nm$^{-1}$'+'\n l = ' + str(lq)[:5] + 'nm')
plt.xlim([3e-2, 5])
plt.ylim([0.9*np.min(radial_profile[~np.isnan(radial_profile)]),
		  1.1*np.max(radial_profile[~np.isnan(radial_profile)])])
plt.xlabel('q (1/m)')
plt.ylabel ('I(q) (a.u.)')
plt.title('...'+file_sample[-40:-4])
plt.legend()
plt.show()

