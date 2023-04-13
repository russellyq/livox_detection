import numpy as np
from filterpy.kalman import KalmanFilter


class Covariance(object):
	'''
	Define different Kalman Filter covariance matrix
	Kalman Filter states:
	[x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
	'''
	def __init__(self):
		self.num_states = 11 # with angular velocity
		self.num_observations = 7
		self.P = np.eye(self.num_states)
		self.Q = np.eye(self.num_states)
		self.R = np.eye(self.num_observations)
		self.P[0,0] = 0.01969623
		self.P[1,1] = 0.01179107
		self.P[2,2] = 0.04189842
		self.P[3,3] = 0.52534431
		self.P[4,4] = 0.11816206
		self.P[5,5] = 0.00983173
		self.P[6,6] = 0.01602004
		self.P[7,7] = 0.01334779
		self.P[8,8] = 0.00389245 
		self.P[9,9] = 0.01837525
		self.Q[0,0] = 2.94827444e-03
		self.Q[1,1] = 2.18784125e-03
		self.Q[2,2] = 6.85044585e-03
		self.Q[3,3] = 1.10964054e-01
		self.Q[4,4] = 0
		self.Q[5,5] = 0
		self.Q[6,6] = 0
		self.Q[7,7] = 2.94827444e-03
		self.Q[8,8] = 2.18784125e-03
		self.Q[9,9] = 6.85044585e-03
		self.R[0,0] = 0.01969623
		self.R[1,1] = 0.01179107
		self.R[2,2] = 0.04189842
		self.R[3,3] = 0.52534431
		self.R[4,4] = 0.11816206
		self.R[5,5] = 0.00983173
		self.R[6,6] = 0.01602004

class KalmanBoxTracker(object):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self, bbox3D, info):
		"""
		Initialises a tracker using initial bounding box.
		"""
		# with angular velocity
		self.kf = KalmanFilter(dim_x=11, dim_z=7)       
		self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
		                      [0,1,0,0,0,0,0,0,1,0,0],
		                      [0,0,1,0,0,0,0,0,0,1,0],
		                      [0,0,0,1,0,0,0,0,0,0,1],  
		                      [0,0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,0,1,0,0,0],
		                      [0,0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,0,1]])     

		self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
		                      [0,1,0,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0,0]])

		covariance = Covariance()
		self.kf.P = covariance.P
		self.kf.Q = covariance.Q
		self.kf.R = covariance.R

		self.kf.x[:7] = bbox3D.reshape((7, 1))

		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.hits = 1           # number of total hits including the first detection
		self.hit_streak = 1     # number of continuing hit considering the first detection
		self.first_continuing_hit = 1
		self.still_first = True
		self.age = 0
		self.info = info        # other info associated

	def update(self, bbox3D, info): 
		""" 
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1          # number of continuing hit
		if self.still_first:
			self.first_continuing_hit += 1      # number of continuing hit in the fist time

		######################### orientation correction
		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		new_theta = bbox3D[3]
		if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
		if new_theta < -np.pi: new_theta += np.pi * 2
		bbox3D[3] = new_theta

		predicted_theta = self.kf.x[3]
		if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
			self.kf.x[3] += np.pi       
			if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
			if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
			if new_theta > 0: self.kf.x[3] += np.pi * 2
			else: self.kf.x[3] -= np.pi * 2

		#########################     # flip

		self.kf.update(bbox3D)

		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
		self.info = info
		# getting back speed information
		# self.info[2] = self.kf.x[7, 0]
		# self.info[3] = self.kf.x[8, 0]
		# self.info[4] = self.kf.x[9, 0]


	def predict(self):       
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		self.kf.predict()      
		if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
		if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

		self.age += 1
		if (self.time_since_update > 0):
			self.hit_streak = 0
			self.still_first = False
		self.time_since_update += 1
		self.history.append(self.kf.x)
		return self.history[-1]

	def get_state(self):
		#print(self.kf.x)
		"""
		Returns the current bounding box estimate.
		"""
		return self.kf.x[:7].reshape((7, ))