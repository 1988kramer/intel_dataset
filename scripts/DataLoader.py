import numpy as np
import math
from itertools import izip

class Measurement:

	# initializes the measurement object
	# params:
	#   ranges:  list of 180 raw ranges for each laser beam
	#   odo_x:   x translation from previous pose in robot's local frame
	#   odo_y:   y translation from previous pose in robot's local frame
	#   odo_phi: yaw rotation in from previous pose in robot's local frame
	#            (phi = 0 aligns with the positive x axis)
	def __init__(self, ranges, odo_x, odo_y, odo_phi):
		self.odometry = self.OdoToMat(odo_x, odo_y, odo_phi) # store odometry measurements as 
															 # transformation matrix
		self.ranges = ranges					# store raw ranges unchanged
		self.points = self.RangeToPCL(ranges)	# convert raw ranges to cartesian points

	# converts raw list of ranges to a list of cartesian points
	# (unsure at this point of the transformation between the robot frame and the laser,
	#  robot frame seems to be x forward y left, need to do some calibration here,
	#  will assume for now that the laser scans range from the +y to -y axis in the robot frame)
	# params:
	#   ranges: list of 180 raw ranges for each laser beam
	# returns:
	#   points: list of homogeneous cartesian points in the x-y plane
	def RangeToPCL(self, ranges):

		beam_angle_increment = math.pi / 180.0
		beam_angle = -math.pi / 2.0

		# iterate over list of ranges, converting each from polar to 
		# cartesian coordinates
		points = []
		for beam_length in ranges:

			# only convert points with nonzero range
			if beam_length > 0.05:

				# convert polar to cartesian coordinates
				point_x = beam_length * math.cos(beam_angle)
				point_y = beam_length * math.sin(beam_angle)

				# store x and y values in a numpy array and append it to the point list
				point = np.array([point_x,point_y,1.0])
				points.append(point)

			# increment the beam angle for the next point
			beam_angle += beam_angle_increment

		return np.array(points)

	# converts relative pose in x, y and phi to a transformation matrix
	# params:
	#   odo_x:   float, x translation in robot frame
	#   odo_y:   float, y translation in robot frame
	#   odo_phi: float, rotation in robot frame
	# return:
	#   T: 3x3 transformation matrix
	def OdoToMat(self, odo_x, odo_y, odo_phi):
		T = np.array([[math.cos(odo_phi), -math.sin(odo_phi), odo_x],
					  [math.sin(odo_phi),  math.cos(odo_phi), odo_y],
					  [0,0,1]])
		return T

class DataLoader:

	# loads sensor data from files
	# params:
	#   laser_file_name: txt file containing 1 laser scan per line
	#   odo_file_name:   txt file containing relative movement in robot frame between poses
	#                    1 relative pose per line: x, y, and phi
	def __init__(self, laser_file_name, odo_file_name):

		# initialize measurement list
		self.measurements = []

		# open files and iterate over them line-by-line
		# for the intel dataset the measurements are guaranteed to be
		# temporally aligned, so no timestamp alignment is required here
		laser_file = open(laser_file_name, 'r')
		odo_file = open(odo_file_name, 'r')
		for laser_line, odo_line in izip(laser_file,odo_file):

			# process laser measurements from string to list of floats
			laser_list = laser_line.split(' ')
			ranges = []
			for range_str in laser_list:
				if range_str.strip() != '':
					ranges.append(float(range_str.strip()))

			# process odometry measurement from string to three floats
			odo_list = odo_line.split(' ')
			odo_x = float(odo_list[0].strip())
			odo_y = float(odo_list[1].strip())
			odo_phi = float(odo_list[2].strip())

			# dump raw measurements into a measurement struct for convenience
			new_measurement = Measurement(ranges, odo_x, odo_y, odo_phi)

			self.measurements.append(new_measurement)

		laser_file.close()
		odo_file.close()
