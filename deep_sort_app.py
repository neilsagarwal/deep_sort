# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import pickle
import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

detections_cache = None

def gather_sequence_info(seqinfo):
    return pickle.load(open(seqinfo, "rb"))

def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(seqinfofile, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, max_age, n_init, max_iou_distance, detections_cache_temp):
    
    global detections_cache
    if not os.path.exists(detections_cache_temp):
        os.mkdir(detections_cache_temp)
    detections_cache = detections_cache_temp + "/%s"


    if os.path.exists(output_file):
        print("skipped this experiment")
        return
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(seqinfofile)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric=metric, max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance)
    results = []

    def frame_callback(vis, frame_idx):
        if frame_idx % 1000 == 0:
            print("Processing frame %05d" % frame_idx)

        detections = None
        if os.path.exists(detections_cache % frame_idx):
            detections = pickle.load( open( detections_cache % frame_idx, "rb" ) )
        else:
            # Load image and generate detections.
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)


            detections = [d for d in detections if d.confidence >= min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            pickle.dump(detections, open( detections_cache % frame_idx, "wb" ))
            print ("dumped")



        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        # if display:
        #     image = cv2.imread(
        #         seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
        #     vis.set_image(image.copy())
        #     vis.draw_detections(detections)
        #     vis.draw_trackers(tracker.tracks)

        # Store results.
        # print(len(tracker.tracks), print(len(detections)))
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])



    # Run tracker.
    visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.7, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--max_age", type=int, default=30)
    parser.add_argument(
        "--n_init", type=int, default=3)
    parser.add_argument(
	"--max_iou_distance", type=float, default=0.7)
    parser.add_argument(
        "--detections_cache", type=str)
    parser.add_argument(
        "--vidfolder", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vidfolder = args.vidfolder
    seqinfofile = vidfolder + "/seqinfo"

    detections_cache = vidfolder + "/cached_detections"

    output_file = vidfolder + "/experiments/deep_sort/labels_cosdist%.3f_maxage%d_ninit%d_ioudist%.3f.csv" % (
        args.max_cosine_distance, args.max_age, args.n_init, args.max_iou_distance)

    run(seqinfofile, output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.max_age, args.n_init, args.max_iou_distance, detections_cache)
