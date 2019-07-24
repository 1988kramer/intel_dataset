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
disc = 0.5
matcher = SDFScanMatcher(discretization=disc)
matcher.AddScan(measurements[0].points)
matcher.AddScan(measurements[1].points)
#matcher.AddScan(measurements[2].points)
res,J,grads = matcher.GetResidualAndJacobian(measurements[0].points,np.identity(3))
#sdf = SDFMap([10,10])
print("residual on initial map: {:f}".format(np.linalg.norm(res**2)))

fig = plt.figure()

plt.imshow(matcher.map.map, interpolation='none',vmin=-1.5,vmax=1.5)
plt.grid(True, 'major')
plt.colorbar()

map_space_points = measurements[0].points / disc
map_space_points[:,0] += matcher.map.offsets[0]
map_space_points[:,1] += matcher.map.offsets[1]
plt.scatter(map_space_points[:,1],map_space_points[:,0], c='r', s=0.5)
#plt.scatter(map_space_points[:,1],map_space_points[:,0], c='b', s=0.5)


for i in range(0,map_space_points.shape[0]):
	plt.arrow(map_space_points[i,1],map_space_points[i,0],grads[i,1]*res[i,0],grads[i,0]*res[i,0],width=1.0e-4,color='r')

fig2 = plt.figure()
plt.imshow(matcher.map.priorities, interpolation='none',vmin=0,vmax=10)
'''
def animate(i):
	global pose

	scan = measurements[i].points

	#sdf.UpdateMap(scan,np.identity(3))

	matcher.AddScan(scan)

	plt.clf()
	plt.imshow(matcher.map.map, interpolation='none',vmin=-3.0,vmax=3.0)
	plt.colorbar()



ani = animation.FuncAnimation(fig, animate, range(len(measurements)), interval=3000)
'''

plt.show()