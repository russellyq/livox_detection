import numpy as np
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, iou3d
from AB3DMOT_libs.mahalanobis_kalman_filter import KalmanBoxTracker

def angle_in_range(angle):
	'''
	Input angle: -2pi ~ 2pi
	Output angle: -pi ~ pi
	'''
	if angle > np.pi:
		angle -= 2 * np.pi
	if angle < -np.pi:
		angle += 2 * np.pi
	return angle

def diff_orientation_correction(det, trk):
	'''
	return the angle diff = det - trk
	if angle diff > 90 or < -90, rotate trk and update the angle diff
	'''
	diff = det - trk
	diff = angle_in_range(diff)
	if diff > np.pi / 2:
		diff -= np.pi
	if diff < -np.pi / 2:
		diff += np.pi
	diff = angle_in_range(diff)
	return diff


def greedy_match(distance_matrix):
	'''
	Find the one-to-one matching using greedy allgorithm choosing small distance
	distance_matrix: (num_detections, num_tracks)
	'''
	matched_indices = []

	num_detections, num_tracks = distance_matrix.shape
	distance_1d = distance_matrix.reshape(-1)
	index_1d = np.argsort(distance_1d)
	index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
	detection_id_matches_to_tracking_id = [-1] * num_detections
	tracking_id_matches_to_detection_id = [-1] * num_tracks
	for sort_i in range(index_2d.shape[0]):
		detection_id = int(index_2d[sort_i][0])
		tracking_id = int(index_2d[sort_i][1])
		if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
			tracking_id_matches_to_detection_id[tracking_id] = detection_id
			detection_id_matches_to_tracking_id[detection_id] = tracking_id
			matched_indices.append([detection_id, tracking_id])

	matched_indices = np.array(matched_indices)
	return matched_indices


def associate_detections_to_trackers(detections,trackers, dets=None, trks=None, trks_S=None, mahalanobis_threshold=0.1):
	"""
  	Assigns detections to tracked object (both represented as bounding boxes)

	detections:  N x 8 x 3
	trackers:    M x 8 x 3

	dets: N x 7
	trks: M x 7
	trks_S: N x 7 x 7

	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	if(len(trackers)==0):
		return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
	iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
	distance_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

	assert(dets is not None)
	assert(trks is not None)
	assert(trks_S is not None)
	S_inv = [np.linalg.inv(S_tmp) for S_tmp in trks_S]  # 7 x 7
	S_inv_diag = [S_inv_tmp.diagonal() for S_inv_tmp in S_inv]# 7

	for d,det in enumerate(detections):
		for t,trk in enumerate(trackers):
			S_inv = np.linalg.inv(trks_S[t]) # 7 x 7
			diff = np.expand_dims(dets[d] - trks[t], axis=1) # 7 x 1
			# manual reversed angle by 180 when diff > 90 or < -90 degree
			corrected_angle_diff = diff_orientation_correction(dets[d][3], trks[t][3])
			diff[3] = corrected_angle_diff
			distance_matrix[d, t] = np.sqrt(np.matmul(np.matmul(diff.T, S_inv), diff)[0][0])

	matched_indices = greedy_match(distance_matrix)


	unmatched_detections = []
	for d,det in enumerate(detections):
		if(d not in matched_indices[:,0]):
			unmatched_detections.append(d)
	unmatched_trackers = []
	for t,trk in enumerate(trackers):
		if len(matched_indices) == 0 or (t not in matched_indices[:,1]):
			unmatched_trackers.append(t)

	#filter out matched with low IOU
	matches = []
	for m in matched_indices:
		match = True
		if distance_matrix[m[0],m[1]] > mahalanobis_threshold:
			match = False
		if not match:
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1,2))
	if(len(matches)==0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):
	def __init__(self,covariance_id=2, max_age=2,min_hits=3):
		"""              
		observation: 
		before reorder: [h, w, l, x, y, z, rot_y]
		after reorder:  [x, y, z, rot_y, l, w, h]
		state:
		[x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
		"""
		self.max_age = max_age
		self.min_hits = min_hits
		self.trackers = []
		self.frame_count = 0
		self.reorder = [3, 4, 5, 6, 2, 1, 0]
		self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
		self.covariance_id = covariance_id

	def update(self,dets_all, match_threshold=11):
		"""
		Params:
		dets_all: dict
			dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
			info: a array of other info for each det
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
	
		dets = dets[:, self.reorder]
		self.frame_count += 1

		trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
		to_del = []
		ret = []
		for t,trk in enumerate(trks):
			pos = self.trackers[t].predict().reshape((-1, 1))
			trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
			if(np.any(np.isnan(pos))):
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
		for t in reversed(to_del):
			self.trackers.pop(t)

		dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
		if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		else: dets_8corner = []

		trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
		trks_S = [np.matmul(np.matmul(tracker.kf.H, tracker.kf.P), tracker.kf.H.T) + tracker.kf.R for tracker in self.trackers]

		if len(trks_8corner) > 0: 
			trks_8corner = np.stack(trks_8corner, axis=0)
			trks_S = np.stack(trks_S, axis=0)

		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner, dets=dets, trks=trks, trks_S=trks_S, mahalanobis_threshold=match_threshold)
	
		#update matched trackers with assigned detections
		for t,trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
				trk.update(dets[d,:][0], info[d, :][0])


		#create and initialise new trackers for unmatched detections
		for i in unmatched_dets:        # a scalar of index
			trk = KalmanBoxTracker(dets[i,:], info[i, :]) 
			self.trackers.append(trk)
		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()      # bbox location
			d = d[self.reorder_back]

			if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
				ret.append(np.concatenate((d, [trk.id+1], trk.info)).reshape(1,-1)) # +1 as MOT benchmark requires positive
			i -= 1
			#remove dead tracklet
			if(trk.time_since_update >= self.max_age):
				self.trackers.pop(i)
		if(len(ret)>0): return np.concatenate(ret)      # x, y, z, theta, l, w, h, ID, other info, confidence
		return np.empty((0,15))      