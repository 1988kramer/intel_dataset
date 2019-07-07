'''
SDFMap.py
 - Andrew Kramer

 Stores a 2 Dimensional map of an area as a signed distance function.
 Provides functions to access and update the map based on work in

    Fossel, Joscha-David & Tuyls, Karl & Sturm, Jurgen. (2015). 
    2D-SDF-SLAM: A signed distance function based SLAM frontend 
    for laser scanners. 1949-1955. 10.1109/IROS.2015.7353633. 

'''

import numpy as np
import math
import scipy.odr

class SDFMap:

	# initializes the necessary data structures and parameters
	#
	# params:
	# - size:           tuple, stores the starting spatial extents of the map 
	#                   in the x and y directions in meters
	# - discretization: float, the edge length of a single map cell in meters
	# - k:              max radius in which to update vertices in meters
	#
	def __init__(self, size, discretization=0.5, k=1.0):
		self.k = k
		self.num_x_cells = int(size[0] / discretization)
		self.num_y_cells = int(size[1] / discretization)
		self.map = np.zeros((self.num_x_cells,self.num_y_cells))


	# updates the map using the given laser scan
	# assumes new scan falls completely within the map bounds
	# need to add an "expand map" function to correct this
	#
	# params:
	# - scan: list of scan endpoints in the robot's local coordinate frame
	#         scan endpoints represented as homogeneous 2D cartesian points
	# - pose: the robot's estimated pose at the time of the scan
	#         represented as a 3x3 transformation matrix
	def UpdateMap(self, scan, pose):

		# first transform scan endpoints to global frame
		global_scan = np.dot(scan, pose.T)

		# find groups of points that occupy the same map cell
		point_groups = self.GroupPointsByCell(global_scan)

		# generate map updates for each cell group
		for group in point_groups:

			A,b = self.LinearFit(group, pose)

			vertices = self.GetUpdateVertices(A, group[0])

			updates = self.GetDistAndPriority(vertices, A, b, pose)


	# calculates the shortest distance to an obstacle and the update priority
	# for the given list of vertices
	# params:
	# - vertices: list of vertices for which to calculate updates
	# - A:        slope of the line fitted to the scan endpoints in the cell
	# - b:        y intercept of the line fitted to the scan endpoints in the cell
	# - pose:     pose of the robot
	# returns:
	# - updates:    list of updated distances for each vertex
	# - priorities: list of priorities for each update
	def GetDistAndPriority(vertices, A, b, pose):



	# get list of vertex indices to update per requirements 
	# in section III.A of Fossel et al
	# params:
	# - A:      slope of line fitted to points in cell
	# - point:  a point inside the cell
	# returns:
	# - indices: list of tuples; vertex indices to be updated
	def GetUpdateVertices(self, A, point):

		# get vertices bounding the current cell
		x_min,x_max,y_min,y_max = self.GetBoundingVertices(point)
		x_c = (x_max + x_min) / 2.0
		y_c = (y_max + y_min) / 2.0

		# get lines bounding the cells to update
		A_prime = -1.0 / A # slope of both bounding lines
		b_lower = 0.0 # y intercept of lower line in cells (not meters)
		b_upper = 0.0 # y intercept of upper line in cells (not meters)
		if A_prime < 0:
			b_lower = y_min - (A_prime * x_max)
			b_upper = y_max - (A_prime * x_min)
		else:
			b_lower = y_min - (A_prime * x_min)
			b_upper = y_max - (A_prime * x_max)

		# search nearby vertices and create list of vertices to update
		# assumes updated cells never go off the edge of the map
		# need to make function to expand map in this case
		indices = []
		k_cells = int(self.k / self.discretization)
		x_range_min = int(x_c - k_cells)
		x_range_max = int(x_c + k_cells)
		y_range_min = int(y_c - k_cells)
		y_range_max = int(y_c + k_cells)
		for x in range(x_range_min, x_range_max):
			for y in range(y_range_min, y_range_max):

				# rule out vertices based on distance to cell center
				dist = math.sqrt((x - x_c)**2 + (y - y_c)**2)
				if dist >= k_cells:
					continue

				# rule out cells not between the two bounding lines
				Ax = A_prime * x
				if Ax + b_upper < y or Ax + b_lower > y:
					continue

				indices.append((x,y))

		return indices


	# fits a line to the given list of points using orthogonal regression
	# if the list contains only one point, the line is perpendicular to 
	# the line between that point and the robot's current point
	# params:
	# - points: group of points in the global frame that occupy the same map cell
	# - pose:   robot's current pose expressed as a transformation matrix
	# returns:
	# - A: the slope of the fitted line
	# - b: the y intercept of the fitted line in meters
	def LinearFit(self, points, pose):
		if len(points) == 1:
			# get perpendicular fit
			A = -1.0 * (points[0][0] - pose[0][2]) / (points[0][1] - pose[1][2])
			b = points[0][1] - (A * points[0][0])
		else:
			# get fit from orthogonal regression (thanks scipy!)
			points = np.array(points)
			data = scipy.odr.RealData(points[:,0],points[:,1])
			odr = ODR(data, scipy.odr.polynomial(1))
			output = odr.run()
			b = output.beta[0]
			A = output.beta[1]

		return A,b

	# groups scan endpoints that fall into the same map cell
	# params:
	# - points:       list of scan endpoints expressed in the global frame
	# returns:
	# - point groups: list of lists of points, each list of points falls into the
	#                 same map cell
	def GroupPointsByCell(self, points):

		# iterate over all scan endpoints, finding groups that occupy the same map cell
		point_groups = []
		point_idx = 0
		while point_idx < len(points):

			cur_point = points[point_idx]

			x_min,x_max,y_min,y_max = self.GetBoundingVertices(cur_point)

			# find all other scan endpoints that fall within the same cell
			same_cell = True
			cur_group = [cur_point]
			point_idx += 1
			while same_cell and point_idx < len(global_scan):
				
				next_point = points[point_idx]

				next_x = next_point[0] / self.discretization
				next_y = next_point[1] / self.discretization

				if (next_x <= x_max and next_x > x_min and 
					next_y <= y_max and next_y > y_min):
					cur_group.append(next_point)
				else:
					same_cell = False

				point_idx += 1

			point_groups.append(cur_group)

		return point_groups


	# gets the map vertices on either side of the given point
	# params:
	# - point: point in the global frame
	# returns:
	# - x_min: map vertex index immediately below the given point in x
	# - x_max: map vertex index immediately above the given point in x
	# - y_min: map vertex index immediately below the given point in y
	# - y_max: map vertex index immediately above the given point in y
	def GetBoundingVertices(self, point):

		# find four grid vertices that bound the current point
		x_min = math.floor(point[0] / self.discretization)
		x_max = x_min_idx + 1.0
		y_min = math.floor(point[1] / self.discretization)
		y_max = y_min_idx + 1.0

		return x_min,x_max,y_min,y_max
