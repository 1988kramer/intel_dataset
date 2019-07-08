'''
SDFScanMatcher.py
 - Andrew Kramer

Creates an SDF map from inputted 2 dimensional laser scans based 
on the method presented in:

    Fossel, Joscha-David & Tuyls, Karl & Sturm, Jurgen. (2015). 
    2D-SDF-SLAM: A signed distance function based SLAM frontend 
    for laser scanners. 1949-1955. 10.1109/IROS.2015.7353633. 

Finds the best alignment between new scans and the map using 
Gauss-Newton optimization, then updates the map with the aligned scan.

'''

import numpy as np 
import math
from SDFMap import SDFMap

class SDFScanMatcher:

	# initializes the SDF map and sets its initial size
	def __init__(self, init_pose=np.identity(3), init_size=(20,20), discretization=0.5, k=2.0):
		self.map = SDFMap(init_size, discretization, k)
		self.pose = init_pose

	# finds the best alignment between the inputted scan and the map
	# adds the new scan to the map once an alignment is found
	# params:
	# - scan: the new scan's endpoints in the robot's frame of reference
	#		  2D numpy array with one point per row 
	#         (expressed in homogeneous coordinates)
	# - pose_delta_guess: an initial guess of the change in the robot's 
	#                     pose since the last scan
	# - max_iter: int, maximum number of Gauss-Newton iterations to use
	# - min_d_err: float, minimum change in error since last iteration to continue iterating
	def AddScan(self, scan, pose_delta_guess=np.identity(3), max_iter=10, min_d_err=1.0e-4):
		
		err = 1.0e4
		d_err = 1.0e4
		num_iter = 0

		next_pose = np.dot(self.pose,pose_delta_guess)

		while num_iter < max_iter and d_err > min_d_err:

			# transform points to global frame
			gf_scan = np.dot(scan,next_pose.T)

			# get map values and gradients for every scan endpoint
			vals = np.zeros(scan.shape[0])
			grads = np.zeros((scan.shape[0],2))
			for scan_idx in range(scan.shape[0]):
				val,grad = self.map.GetMapValueAndGradient(gf_scan[scan_idx,:2])
				vals[scan_idx] = val
				grads[scan_idx,:] = grad

			# assemble measurement and jacobian matrices

			# calculate the pose update

			# evaluate the error


			num_iter += 1
