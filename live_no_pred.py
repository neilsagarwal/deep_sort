# pip install opencv-contrib-python
import cv2
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import time
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
import shelve

from deep_sort import tools.generate_detections.create_box_encoder

db = shelve.open("db")

# iterate through frames... skip based on param
WINDOW_NAME = "COCO detections"
video_path = "../../../videosec/dp_surveillance_data/videos/prev_videos/auburn_short/auburn_short.mp4"
frame_interval = 8
cpu_device = torch.device("cpu")
config_file = "detectron2/configs/quick_schedules/mask_rcnn_R_50_DC5_inference_acc_test.yaml"
threshold = 0.5

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
    instances = predictions["instances"]

    print("Detected {} instances in {:.2f}s".format(
        len(predictions["instances"]), time.time() - start_time
    ))

    visualizer = Visualizer(im, metadata, instance_mode=ColorMode.IMAGE)
    instances = predictions["instances"].to(cpu_device)

    classification_mask = [class_idx == index_of_person and score > threshold
        for class_idx, score in zip(instances.pred_classes, instances.scores)]

    masked_instances = Instances(instances.image_size)
    masked_instances.set('pred_boxes', instances.pred_boxes[classification_mask])
    masked_instances.set('scores', instances.scores[classification_mask])
    visualized_output = visualizer.draw_instance_predictions(predictions=masked_instances)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
