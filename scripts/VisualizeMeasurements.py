import numpy as np 
from DataLoader import Measurement, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

parser = argparse.ArgumentParser(description='visualizer for raw measurements from the intel dataset')
parser.add_argument('--laser_file', type=str, default='../data/intel_LASER_.txt',
					help='name of the laser scanner log file')
parser.add_argument('--odometry_file', type=str, default='../data/intel_ODO.txt',
					help='name of the odometry log file')
args = parser.parse_args()

loader = DataLoader(args.laser_file, args.odometry_file)
measurements = loader.measurements

fig = plt.figure()

axes_points = np.array([[0,1,0],
					    [0,0,1],
					    [1,1,1]])

abs_pose = np.identity(3)

def animate(i):
	global abs_pose

	# transform laser measurements to global frame
	in_points = np.array(measurements[i].points)
	tf_points  = np.dot(abs_pose,np.transpose(in_points))
	tf_points = np.transpose(tf_points)

	# update global pose from odometry
	abs_pose = np.dot(abs_pose, measurements[i].odometry)
	tf_axes = np.dot(abs_pose, axes_points)

	plt.clf()
	plt.xlim(xmax = 20, xmin = -20)
	plt.ylim(ymax = 20, ymin = -20)

	# plot robot pose
	plt.plot(tf_axes[0,:2],tf_axes[1,:2],c='r')
	plt.plot(tf_axes[0,:3:2],tf_axes[1,:3:2],c='g')

	# plot laser measurements
	plt.scatter(tf_points[:,0],tf_points[:,1],marker='.',s=1.0)


ani = animation.FuncAnimation(fig, animate, len(measurements), interval=100)
plt.show()