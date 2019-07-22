import numpy as np 
from DataLoader import Measurement, DataLoader
from SDFMap import SDFMap
from SDFScanMatcher import SDFScanMatcher
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

matcher = SDFScanMatcher()
#sdf = SDFMap([10,10])

fig = plt.figure()


def animate(i):
	global pose

	scan = measurements[i].points

	#sdf.UpdateMap(scan,np.identity(3))

	matcher.AddScan(scan)

	plt.clf()
	plt.imshow(matcher.map.map, interpolation='none',vmin=-3.0,vmax=3.0)
	plt.colorbar()



ani = animation.FuncAnimation(fig, animate, range(len(measurements)), interval=3000)


plt.show()