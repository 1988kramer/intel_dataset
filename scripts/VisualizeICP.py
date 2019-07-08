import numpy as np 
from DataLoader import Measurement, DataLoader
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from Align2D import Align2D

parser = argparse.ArgumentParser(description='visualizer for raw measurements from the intel dataset')
parser.add_argument('--laser_file', type=str, default='../data/intel_LASER_.txt',
					help='name of the laser scanner log file')
parser.add_argument('--odometry_file', type=str, default='../data/intel_ODO.txt',
					help='name of the odometry log file')
args = parser.parse_args()

loader = DataLoader(args.laser_file, args.odometry_file)
measurements = loader.measurements

axes_points = np.array([[0,1,0],
					    [0,0,1],
					    [1,1,1]])

fig = plt.figure()

pose = np.identity(3)

def animate(i):
	global pose

	# transform target points to current estimated pose
	target_points = measurements[i].points
	source_points = measurements[i+1].points

	# initial guess at transform is identity
	initial_T = np.identity(3) 
	#initial_T = measurements[i+1].odometry

	# find transform between point clouds via ICP
	aligner = Align2D(source_points, target_points, initial_T)
	est_T = aligner.transform

	# align target points to global frame
	target_points = np.dot(target_points, pose.T)
	# update pose
	pose = np.dot(pose, est_T)
	# align source points to target frame
	source_points = np.dot(source_points, pose.T)

	tf_axes = np.dot(pose, axes_points)

	plt.clf()
	plt.xlim(xmax = 25, xmin = -5)
	plt.ylim(ymax = 10, ymin = -20)

	# plot robot pose
	plt.plot(tf_axes[0,:2],tf_axes[1,:2],c='r')
	plt.plot(tf_axes[0,:3:2],tf_axes[1,:3:2],c='g')

	# plot source and target point clouds
	plt.scatter(target_points[:,0],target_points[:,1],marker='.',s=1.0,c='c')
	plt.scatter(source_points[:,0],source_points[:,1],marker=',',s=1.0,c='m')



ani = animation.FuncAnimation(fig, animate, len(measurements)-1, interval=1)
plt.show()