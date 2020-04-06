"""
Group of functions to build the matrices used in the analysis of SAXS data
- distance matrix: coordinates in the detector plane indicating the distance from the beam center
- q values matrix: spatial wave vector qx, qy, coordinates from scattering


Notes:
- both the distance and qvalue matrices depend only on the shape of the detector/image,
and the calibration parameters. Thus, it is necessary to build them only once per experiment,
and not for each image

Created on Fri Nov 24 2017
Last modification on Fri March 2nd, 2018
version: 0.0

@author: Cristina MT
"""

class BuildMatrix():
	"""
	Build the different 'coordinate' matrices for the analysis of SAXS data
	- distance matrix: 2D array, same dimensions as image
	- q values matrix: 2D array, same dimensions as image
	"""

	def distance(shape_mat, cal_info, center = None):
		"""
		Defines function that builds the distance matrix, in [m]

		INPUT:
		[shape_mat] 1D array, containing the xy dimensions [ydim, xdim] of the saxs image
		[cal_info] dictionary containing the calibration information from poni file,
		it is obtained with the function OpenSAXS.read_poni() in the file saxs_open.py

		OUTPUT:
		[ycenter, xcenter] array with coordinates (in pixels) of the beam center
		[mat_dist] distance matrix as a 2D array
		"""

		import numpy as np

		mat_dist = np.zeros(shape_mat)

		# Build grid for x and y direction
		x = np.arange(shape_mat[1])
		y = np.arange(shape_mat[0])
		xg = np.tile(x, (len(y),1)); yg = np.tile(y, (len(x), 1)).transpose()

		if center is None:
			# Get center from calibration info in poni file
			xcenter = np.float(cal_info.get('Poni2'))/np.float(cal_info.get('PixelSize2'))
			ycenter = np.float(cal_info.get('Poni1'))/np.float(cal_info.get('PixelSize1'))
		else:
			xcenter = center[1]
			ycenter = center[0]

		# Build distance matrix
		mat_dist = np.sqrt((xg - xcenter)**2+(yg -ycenter)**2)  # in pixels
		cal_pix = np.float(cal_info.get('PixelSize1'))  # pixel calibration factor
		mat_dist = mat_dist*cal_pix						# distance in [m]

		return [ycenter, xcenter], mat_dist

	def qvalues(mat_dist, cal_info):
		"""
		Defines function that builds the 2D array [mat_qvalues] of the matrix of q values,
		in [nm^(-1)]

		INPUT:
		[mat_dist] 2D array containing the values of the distance matrix (in m),
		it is obtained with the function BuildMatrix.distance()
		[cal_info] dictionary containing the calibration information from poni file,
		it is obtained with the function OpenSAXS.read_poni() in the file saxs_open.py
		"""

		import numpy as np

		dist_det = np.float(cal_info.get('Distance'))  # Distance detector
		wlambda = np.float(cal_info.get('Wavelength'))
		theta = np.arctan(mat_dist/dist_det)/2			# Angle respect to the detector
		mat_qvalues = 4*np.pi*np.sin(theta)/wlambda *(10**(-9))  # q matrix in [nm^-1]

		return mat_qvalues
