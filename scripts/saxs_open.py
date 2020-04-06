"""
Group of functions to read and display raw data generated from SAXS experiments at the ESRF
- poni file: calibration information
- mask file: identifies the detector lines and beam stopper
- edf file: contains the saxs data

Notes:
- Mask is read either from txt file generated with matlab code, or edf file.
- EDF file is read in the simplest form possible. Adapted from the matlab code by the saxs team.
It's possible it doesn't read all the files. If this is true, more options need to be added

Created on Fri Nov 24 2017
Last modification on Mon Jan 8 2018
version: 0.0

@author: Cristina MT
"""

class OpenSAXS():
	"""
	Opens the different types of data generated from SAXS experiments:
	- poni file (.poni) : can be read in a text editor
	- mask file (.msk): binary file, in bits
	- edf file (.edf) : header text + data in binary format, bytes
	"""

	def read_poni(filename):
		"""
		Defines function to read the poni file whose name is [filename]
		[filename] is a string variable containing the full path of the file,
		including the file extension

		Output: [cal_info] dictionary containing the calibration information
		Typically, it contains the following keys:
		- PixelSize1 , PixelSize2
		- Distance
		- Poni1 , Poni2
		- Rot1 , Rot2 , Rot3 (not used)
		- Wavelength
		Note: 1 refers to X axis, 2 refers to Y axis
		"""

		cal_info = dict()
		try:
			with open(filename, 'r') as f:
				file_data = f.readlines()  	# Read and store all file content at once
			for line in file_data:
				if line[0] != '#':			# Discard the comment lines
					ind_sep = line.find(':') # ':' delimits the key name, and its value
					if ind_sep > 0:			# make sure it's a variable, and not blank
						key_name = line[0:ind_sep]
						key_value = line[ind_sep+2:-1]
						cal_info[key_name] = key_value
		except FileNotFoundError: print('Error File Does Not Exist: '+filename)
		return cal_info

	def read_mask_msk(filename):
		"""
		Defines function to read the mask file whose name is [filename]
		[filename] is a string variable containing the full path of the file,
		including the file extension

		Output: [mask] numpy 2D array with 0|1 values

		Warning: do not use! issue with reading file as bits
		- arbitrary offsets? to be checked
		note: adapted from B.Vos CreateMask matlab code
		(matlab does have a bultin function to read bits)
		"""

		import numpy as np

		mask_bytes = np.fromfile(filename, 'uint8')	# smallest info numpy reads is bytes
		mask_bits = np.unpackbits(mask_bytes)		# convert bytes to bits

		# Arbitrary offsets, obtained by trial/error to match SAXS data array shape
		xo = 11	; yo = 8
		offset = (1043+yo)*(981+xo)

		# Reshape the string of bits into a 2D array
		mask = mask_bits[245:245+offset].reshape([1043+yo, 981+xo])[yo:,xo:]

		return mask

	def read_mask_txt(filename):
		"""
		Defines function to read the mask file whose name is [filename]
		[filename] is a string variable containing the full path of the file,
		including the file extension

		Important: Mask should be in txt format

		Output: [mask] numpy 2D array with 0|1 values
		"""

		import numpy as np

		mask_raw = np.loadtxt(filename, delimiter = ',') # mask in txt is saved with xy inverted
		mask = mask_raw.transpose()

		return mask

	def read_mask_edf(filename):
		"""
		Defines function to read the mask file whose name is [filename]
		[filename] is a string variable containing the full path of the file,
		including the file extension

		Important: Mask should be in edf format

		Output: [mask] numpy 2D array with 0|1 values
		"""

		import numpy as np

		mask_bytes = np.fromfile(filename, 'uint8')	# mask is saved as unsigned byte
		offset = 1024  # Arbitrary offset, obtained by trial/error to match SAXS data array shape
		mask = mask_bytes[offset:].reshape(1043, 981)   # Reshape the string of bytes into a 2D array

		return mask

	def read_edf(filename):
		"""
		Defines function to read the edf file whose name is [filename]
		[filename] is a string variable containing the full path of the file,
		including the file extension

		Output:
		[header_info] dictionary with all the information in file header
		[image] numpy 2D array containing the SAXS data
		"""

		import numpy as np
		import codecs

		header_info = dict()
		image = []
		try:
			read_line = 1	# Variable used to stop reading file when header is finished.
			# Reads file line by line to extract information from the header.
			with codecs.open(filename, "r",encoding='utf-8', errors='replace') as f:
				f.readline()
				while read_line == 1:
					line = f.readline()
					ind_sep = line.find('=')  # '=' delimits the key name from its value.
					if ind_sep > 0:			  # make sure there's a variable, and not empty/binary.
						key_name = line[0:ind_sep-1]
						key_value = line[ind_sep+2:-3]
						header_info[key_name] = key_value
					else: read_line = 0       # if '=' is not found, the header is over, stop reading.
			# Read the file into a numpy array.
			if header_info.get('DataType') == 'SignedInteger':  # Condition extracted from matlab code. There are other options available.
				dt = 'int32'
				data = np.fromfile(filename,
									dtype = dt,
									count = int(header_info['Size']))  # Read the file as [dt] type, stop when the [Size] reported in header is reached.
				# Get XY dimensions for the 2D array
				xdim = int(header_info.get('Dim_1'))
				ydim = int(header_info.get('Dim_2'))
				# Reshape the 1D array into correct dimensions. The 512 offset is arbitrary obtained by trial/error
				image = data[512:].reshape(ydim, xdim)
			else:  print('['+header_info.get('DataType')+'] data type not known')  # If the data type is not 'SignedInteger', but header can be read
		except FileNotFoundError: print('Error File Does Not Exist: '+filename)
		return image, header_info

class ShowSAXS():
	"""
	Displays the different types of raw data read with OpenSAXS class:
	- mask: binary array
	- image: SAXS data array

	Note: Each time a function is called, it creates a new figure
	"""

	def raw_img(saxs_image, vmin = None, vmax = None,
				show_center = None, log_scale = None, colormap = None):
		"""
		Defines function that shows the raw SAXS data image contained in the [saxs_image]
		2D numpy array.
		[vmin] is the value minimum for the colorscale. By default is the minimum of the array
		[vmax] same as [vmin] but for the maximum value
		[show_center] is an array [x,y] . If 'None' , it doesn't show the center of the beam
		[log_scale] By default it applies the np.log of SAXS data. Use False to prevent this
		[colormap] By default is 'inferno'. Accepts any other matplotlib colormap name as a string
		"""

		import numpy as np
		import matplotlib.pyplot as plt

		plt.figure()

		# Configure display according to options

		if log_scale == None: image = np.log(saxs_image)
		elif log_scale == False: image = 1.*saxs_image

		if colormap == None: cmap_choice = 'inferno'
		else: cmap_choice = colormap

		if vmin == None: v1 = np.min(image.flatten())
		else: v1 = vmin
		if vmax == None: v2 = np.max(image.flatten())
		else: v2 = vmax

		# Plot image and center, if applicable

		plt.imshow(image,
					cmap = cmap_choice,
					interpolation = 'none',
					vmin = v1, vmax = v2)

		if show_center != None:
			plt.plot([0, image.shape[1]],[show_center[1], show_center[1]], '--', c = 'cyan')
			plt.plot([show_center[0], show_center[0]],[0, image.shape[0]], '--', c = 'cyan')

		plt.xlim([0, image.shape[1]])
		plt.ylim([image.shape[0], 0])
		#plt.show()

	def mask(mask_array, show_center = None):
		"""
		Defines function that shows the [mask_array] 2D numpy array
		used to discard the detector lines and beam stopper.
		[show_center] is an array [x,y] . If 'None' , it doesn't show the center of the beam
		"""

		import matplotlib.pyplot as plt

		# Plot image and center, if applicable
		plt.figure();
		plt.imshow(mask_array,
					cmap = 'gray',
					interpolation = 'none')

		if show_center != None:
			plt.plot([0, mask_array.shape[1]],[show_center[1], show_center[1]], '--', c = 'cyan')
			plt.plot([show_center[0], show_center[0]],[0, mask_array.shape[0]], '--', c = 'cyan')

		plt.xlim([0, mask_array.shape[1]])
		plt.ylim([mask_array.shape[0], 0])
		#plt.show()

	def img_wmask(saxs_image, mask, vmin = None, vmax = None,
				show_center = None, log_scale = None, colormap = None, alpha = None):
		"""
		Combines the function ShowSAXS.raw_img() and ShowSAXS.mask() into one,
		to show the mask overlayed on the image, using [alpha] as transparecy value.
		[alpha] can take values 0 - 1.0. Default is 0.5

		Warning: Display might be too slow and could block python, depending on how is it running
		Function to use carefully
		"""

		import matplotlib.pyplot as plt

		#plt.ion()
		ShowSAXS.raw_img(saxs_image, vmin, vmax, show_center, log_scale, colormap)

		if alpha == None: a_value = 0.5
		else: a_value = alpha

		# manipulate the mask for diaplay purposes
		mask_new = 1.*mask
		mask_new[mask==0] = float('nan')

		plt.imshow(mask_new,
					interpolation = 'none',
					alpha = a_value,
					vmin = 0, vmax = 1,
					cmap = 'Reds')

