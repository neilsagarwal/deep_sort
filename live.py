import cv2
import time
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer

import shelve
import numpy as np

from tools.generate_detections import create_box_encoder
from application_util import preprocessing
from application_util.visualization import create_unique_color_float

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# if True, only look at frames that have more than 3% difference than frames before
# reduces number of frames needed to go through detection algo
flag_trigger = False

flag_tracking_on = True # turn tracking results on; else, show all detections available

flag_save_video = False # store results in a .mp4 in same directory

model = "resources/networks/mars-small128.pb"
encoder = create_box_encoder(model, batch_size=1)

# iterate through frames... skip based on param
WINDOW_NAME = "COCO detections"
video_path = None # MODIFY VIDEO PATH HERE

assert video_path is not None, "Please set video_path"
frame_interval = 8
cpu_device = torch.device("cpu")
config_file = "detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # make sure to change corresponding weights!!
confidence = 0.1

frame_diff_thresh = 35 # pulled from glimpse paper...
new_detection_needed_thresh_percent = 3 # 3 percent of pixels diff to engage.
new_detection_needed_thresh = None # None bc calculated based off first im and then set

dbname = video_path.split("/")[-1].split(".")[0]
db = shelve.open("db_%s" % dbname)
db_dets = shelve.open("db_dets_%s" % dbname)

cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
cfg.freeze()

metadata = MetadataCatalog.get(
    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
)

predictor = DefaultPredictor(cfg)

vdo = cv2.VideoCapture()
vdo.open(video_path)
width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

if flag_save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter('%s.mp4' % dbname, fourcc, 30.0, (width, height), True)

assert vdo.isOpened()

max_cosine_distance = 0.2
nn_budget = None
nms_max_overlap = 1.0
max_iou_distance=1 # default is 0.7
max_age=10 # default is 30
n_init = 0 # default is 3

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

idx_frame = 0

stored_frame_gray = None
stored_colors = None
stored_boxes = None

class_labels = metadata.get("thing_classes")
index_of_person = class_labels.index("person")

skipped_frames = 0

while vdo.grab():
    _, ori_im = vdo.retrieve()
    im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

    if new_detection_needed_thresh is None:
        new_detection_needed_thresh = float(width*height) * new_detection_needed_thresh_percent / 100

    idx_frame += 1
    if idx_frame % frame_interval == 0:

        curr_frame_gray = cv2.cvtColor(ori_im, cv2.COLOR_BGR2GRAY)

        if flag_trigger and stored_frame_gray is not None:
            diff = sum(sum((cv2.absdiff(curr_frame_gray,stored_frame_gray) > frame_diff_thresh).astype(int)))

        if not flag_trigger or stored_frame_gray is None or diff> new_detection_needed_thresh:
            if stored_frame_gray is not None and flag_trigger:
                print("Diff threshold surpassed; diff = %d." % diff)

            # Cache object detection results!
            if str(idx_frame) in db.keys():
                predictions = db[str(idx_frame)]
            else:
                predictions = predictor(im)
                db[str(idx_frame)] = predictions

            stored_frame_gray = curr_frame_gray

            instances = predictions["instances"].to(cpu_device)

            # only look at ppl & confidence > threshold
            classification_mask = [class_idx == index_of_person and score > confidence
                for class_idx, score in zip(instances.pred_classes, instances.scores)]

            boxes = []
            # get width, height
            for box in instances.pred_boxes[classification_mask]:
                boxes += [[box[0], box[1], box[2]-box[0], box[3]-box[1]]]
            scores = instances.scores[classification_mask]

            # Cache featurization... probably shouldn't be labeled as detections
            if str(idx_frame) in db_dets.keys():
                detections_out = db_dets[str(idx_frame)]
            else:
                features = encoder(ori_im, boxes)
                detections_out = [Detection(box, score, feature) for (box, feature, score) in zip(boxes, features,scores)]
                boxes_tlwh = np.array([d.tlwh for d in detections_out])
                indices = preprocessing.non_max_suppression(boxes_tlwh, nms_max_overlap, scores)
                detections_out = [detections_out[i] for i in indices]
                db_dets[str(idx_frame)] = detections_out

            if flag_tracking_on:
                tracker.predict(skipped_frames=skipped_frames)
                tracker.update(detections_out)
                skipped_frames = 1

                colors = []
                new_boxes = []
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue
                    colors += [create_unique_color_float(track.track_id)]
                    new_boxes += [[*track.to_tlbr().astype(np.int)]]
                stored_boxes = new_boxes
                stored_colors = colors
            else:
                stored_boxes = instances.pred_boxes[classification_mask]
                stored_colors = None

        else:
            skipped_frames += 1


    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if stored_boxes is None:
        if flag_save_video:
            out_vid.write(im[:, :, ::-1])
        cv2.imshow(WINDOW_NAME, im[:, :, ::-1])
    else:
        print(len(stored_boxes))
        visualizer = Visualizer(im, metadata, instance_mode=ColorMode.IMAGE)
        visualized_output = visualizer.overlay_instances(boxes=stored_boxes, assigned_colors=stored_colors)
        img = visualized_output.get_image()[:, :, ::-1]
        if flag_save_video:
            out_vid.write(img)
        cv2.imshow(WINDOW_NAME, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        db.close()
        db_dets.close()
        if flag_save_video:
            out_vid.release()
        break
