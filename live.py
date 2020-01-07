# pip install opencv-contrib-python
# python -m pip install tensorflow==1.14

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

model = "resources/networks/mars-small128.pb"
encoder = create_box_encoder(model, batch_size=1)

# iterate through frames... skip based on param
WINDOW_NAME = "COCO detections"
video_path = "../../../videosec/dp_surveillance_data/videos/prev_videos/auburn_short/auburn_short.mp4"
frame_interval = 8
cpu_device = torch.device("cpu")
config_file = "detectron2/configs/quick_schedules/mask_rcnn_R_50_DC5_inference_acc_test.yaml"
threshold = 0.5

dbname = video_path.split("/")[-1].split(".mp4")[0]
print(dbname)
db = shelve.open("db_%s" % dbname)

cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.DEVICE = "cpu"
cfg.freeze()

metadata = MetadataCatalog.get(
    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
)

predictor = DefaultPredictor(cfg)

vdo = cv2.VideoCapture()
vdo.open(video_path)

assert vdo.isOpened()

max_cosine_distance = 0.2
nn_budget = None
nms_max_overlap = 1.0

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)


idx_frame = 0

class_labels = metadata.get("thing_classes")
index_of_person = class_labels.index("person")
while vdo.grab():
    idx_frame += 1
    if idx_frame % frame_interval == 0:
        continue

    start_time = time.time()
    _, ori_im = vdo.retrieve()
    im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

    if str(idx_frame) in db.keys():
        predictions = db[str(idx_frame)]
    else:
        predictions = predictor(im)
        db[str(idx_frame)] = predictions

    instances = predictions["instances"].to(cpu_device)
    classification_mask = [class_idx == index_of_person and score > threshold
        for class_idx, score in zip(instances.pred_classes, instances.scores)]

    boxes = []
    for box in instances.pred_boxes[classification_mask]:
        boxes += [[box[0], box[1], box[2]-box[0], box[3]-box[1]]]
    scores = instances.scores[classification_mask]

    features = encoder(ori_im, boxes)

    detections_out = [Detection(box, score, feature) for (box, feature, score) in zip(boxes, features,scores)]
    boxes = np.array([d.tlwh for d in detections_out])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections_out = [detections_out[i] for i in indices]

    tracker.predict()
    tracker.update(detections_out)

    colors = []
    boxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        colors += [create_unique_color_float(track.track_id)]
        boxes += [[*track.to_tlbr().astype(np.int)]]

    visualizer = Visualizer(im, metadata, instance_mode=ColorMode.IMAGE)
    visualized_output = visualizer.overlay_instances(boxes=boxes, assigned_colors=colors)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
