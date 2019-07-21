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

fig = plt.figure()

pose = np.identity(3)

source_points = measurements[180].points
target_points = measurements[184].points

aligner=Align2D(source_points,target_points,np.identity(3))
matched_trg,matched_src,indices = aligner.FindCorrespondences(source_points)
T = aligner.AlignSVD(matched_src,matched_trg)


plt.xlim(xmax = 15, xmin = -5)
plt.ylim(ymax = 10, ymin = -10)

for i in range(len(matched_trg)):
	x = np.array([matched_trg[i,0],matched_src[i,0]])
	y = np.array([matched_trg[i,1],matched_src[i,1]])
	plt.plot(x,y,c='b',linewidth=0.5)


# plot source and target point clouds
plt.scatter(target_points[:,0],target_points[:,1],marker='.',s=1.0,c='c')
plt.scatter(source_points[:,0],source_points[:,1],marker=',',s=1.0,c='m')



plt.show()

