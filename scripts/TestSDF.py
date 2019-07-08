import numpy as np 
from DataLoader import Measurement, DataLoader
from SDFMap import SDFMap
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

map = SDFMap((25,20))

fig = plt.figure()


pose = np.identity(3)
pose[1][2] = 10.01
pose[0][2] = 5.01

def animate(i):
	global pose

	scan = measurements[i].points

	map.UpdateMap(scan,pose)

	plt.clf()
	plt.imshow(map.map, interpolation='none',vmin=-5.0,vmax=5.0)
	plt.colorbar()



ani = animation.FuncAnimation(fig, animate, 30, interval=1000)


plt.show()