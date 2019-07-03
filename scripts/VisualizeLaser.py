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

def animate(i):
	plt.clf()
	plt.xlim(xmax = 20, xmin = 0)
	plt.ylim(ymax = 10, ymin = -10)
	points = measurements[i].points
	plt.scatter(points[:,0],points[:,1],marker='.',s=1.0)


ani = animation.FuncAnimation(fig, animate, len(measurements), interval=100)
plt.show()
	
