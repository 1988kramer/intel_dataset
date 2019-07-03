import numpy as np 
from scipy.spatial import KDTree

class Align2D:

	# params:
	#   source_points: numpy array containing points to align to the target set
	#                  points should be homogeneous, with one point per row
	#   target_points: numpy array containing points to which the source points
	#                  are to be aligned, points should be homogeneous with one
	#                  point per row
	#   initial_T:     initial estimate of the transform between target and source
	def __init__(self, source_points, target_points, initial_T):
		self.source = source_points
		self.target = target_points
		self.init_T = initial_T
		self.target_tree = KDTree(target_points[:,:2])
		self.transform = self.AlignICP(10, 1.0e-2)

	# uses the iterative closest point algorithm to find the
	# transformation between the source and target point clouds
	# that minimizes the sum of squared errors between nearest 
	# neighbors in the two point clouds
	# params:
	#   max_iter: int, max number of iterations
	#   min_delta_err: float, minimum change in alignment error
	def AlignICP(self, max_iter, min_delta_err):

		sum_sq_error = 1.0e6 # initialize error as large number
		delta_err = 1.0e6    # change in error (used in stopping condition)
		T = self.init_T
		num_iter = 0         # number of iterations
		tf_source = self.source

		while delta_err > min_delta_err and num_iter < max_iter:

			# find correspondences via nearest-neighbor search
			matched_trg_pts = self.FindCorrespondences(tf_source)

			# find alingment between source and corresponding target points via SVD
			# note: svd step doesn't use homogeneous points
			new_T = self.AlignSVD(tf_source[:,:2], matched_trg_pts)

			# update transformation between point sets
			T = np.dot(T,new_T)

			# apply transformation to the source points
			tf_source = np.dot(self.source,T.T)

			# find sum squared error between transformed source points and target points
			pt_diffs = tf_source[:,2] - matched_trg_pts
			new_err = 0
			for diff in pt_diffs:
				new_err += np.dot(diff,diff.T)

			# update error and calculate delta error
			delta_err = abs(sum_sq_error - new_err)
			sum_sq_error = new_err

			max_iter += 1

		return T

	# finds nearest neighbors in the target point for all points
	# in the set of source points
	# currently allows for multiple association but may need to change this
	# params:
	#   src_pts: array of source points for which we will find neighbors
	#            points are assumed to be homogeneous
	# returns:
	#   array of nearest target points to the source points (not homogeneous)
	def FindCorrespondences(self,src_pts):

		# get distances to nearest neighbors and indices of nearest neighbors
		dist,indices = self.target_tree.query(src_pts[:,:2])

		# build array of nearest neighbor target points
		point_list = []
		for idx in indices:
			point_list.append(self.target[idx,:])

		matched_pts = np.array(point_list)

		return matched_pts[:,:2]

	# uses singular value decomposition to find the 
	# transformation from the target to the source point cloud
	# assumes source and target point clounds are ordered such that 
	# corresponding points are at the same indices in each array
	#
	# params:
	#   source: numpy array representing source pointcloud
	#   target: numpy array representing target pointcloud
	# returns:
	#   T: transformation between the two point clouds
	def AlignSVD(self, source, target):

		# first find the centroids of both point clouds
		src_centroid = self.GetCentroid(source)
		trg_centroid = self.GetCentroid(target)

		# get the point clouds in reference to their centroids
		source_centered = source - src_centroid
		target_centered = target - trg_centroid

		# get cross covariance matrix M
		M = np.dot(source_centered,np.dot(target_centered))

		# get singular value decomposition of the cross covariance matrix
		U,W,V_t = np.linalg.svd(M)

		# get rotation between the two point clouds
		R = np.dot(V_t.T,U.T)

		# get the translation (simply the difference between the point clound centroids)
		t = trg_centroid - src_centroid

		# assemble translation and rotation into a transformation matrix
		T = np.identity(3)
		T[:2,2] = np.squeeze(t)
		T[:2,:2] = R

		return T

	def GetCentroid(self, points):
		point_sum = np.sum(points,axis=0)
		return point_sum / float(len(points))