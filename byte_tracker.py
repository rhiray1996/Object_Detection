import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box1, box2):
    """Compute IoU between two boxes in [x1,y1,x2,y2] format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union


class Track:
    """Simple track with constant velocity"""

    def __init__(self, bbox, score, class_id, track_id):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.track_id = int(track_id)
        self.velocity = np.zeros(4, dtype=np.float32)
        self.time_since_update = 0

    def predict(self):
        self.bbox += self.velocity
        self.time_since_update += 1

    def update(self, bbox, score, class_id):
        bbox = np.asarray(bbox, dtype=np.float32)
        self.velocity = bbox - self.bbox
        self.bbox = bbox
        self.score = float(score)
        self.class_id = int(class_id)
        self.time_since_update = 0

    def to_tlbr(self):
        return self.bbox.tolist()


class BYTETracker:
    """Simplified ByteTrack implementation"""

    def __init__(self, track_thresh=0.5, match_thresh=0.3, max_age=30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def _associate(self, track_indices, detections, threshold):
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
        cost = np.zeros((len(track_indices), len(detections)), dtype=np.float32)
        for i, t_idx in enumerate(track_indices):
            t_box = self.tracks[t_idx].to_tlbr()
            for j, d in enumerate(detections):
                cost[i, j] = 1.0 - iou(t_box, d)
        row_ind, col_ind = linear_sum_assignment(cost)
        matches, unmatched_tracks, unmatched_dets = [], [], list(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= 1.0 - threshold:
                matches.append((track_indices[r], c))
                unmatched_dets.remove(c)
            else:
                unmatched_tracks.append(track_indices[r])
        for idx in range(len(track_indices)):
            if idx not in row_ind:
                unmatched_tracks.append(track_indices[idx])
        return matches, unmatched_tracks, unmatched_dets

    def update(self, bboxes, scores, classes):
        if len(bboxes) == 0:
            bboxes = np.empty((0, 4))
            scores = np.empty((0, ))
            classes = []
        else:
            bboxes = np.asarray(bboxes, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
        high_mask = scores >= self.track_thresh
        det_high = bboxes[high_mask]
        score_high = scores[high_mask]
        class_high = [classes[i] for i in range(len(classes)) if high_mask[i]]
        det_low = bboxes[~high_mask]
        score_low = scores[~high_mask]
        class_low = [classes[i] for i in range(len(classes)) if not high_mask[i]]

        for t in self.tracks:
            t.predict()

        track_indices = list(range(len(self.tracks)))
        matches_h, unmatched_tracks, unmatched_high = self._associate(track_indices, det_high, self.match_thresh)
        for t_idx, d_idx in matches_h:
            self.tracks[t_idx].update(det_high[d_idx], score_high[d_idx], class_high[d_idx])

        matches_l, unmatched_tracks, unmatched_low = self._associate(unmatched_tracks, det_low, self.match_thresh)
        for t_idx, d_idx in matches_l:
            self.tracks[t_idx].update(det_low[d_idx], score_low[d_idx], class_low[d_idx])

        for d_idx in unmatched_high:
            self.tracks.append(Track(det_high[d_idx], score_high[d_idx], class_high[d_idx], self.next_id))
            self.next_id += 1

        updated_tracks = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                updated_tracks.append(trk)
        self.tracks = updated_tracks

        outputs = []
        for trk in self.tracks:
            if trk.time_since_update == 0:
                x1, y1, x2, y2 = trk.to_tlbr()
                outputs.append([x1, y1, x2, y2, trk.track_id, trk.class_id, trk.score])
        return outputs
