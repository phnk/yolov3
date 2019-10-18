from train import train
from detect import detect
from utils.utils import plot_results

import torch
import cv2
import datetime
import os

def test_video(video_path, yolo_cfg, data_cfg, weights, data_name):
    print("Generating testing video..")
    with torch.no_grad():
        out_size = (1920, 1080)

        # works on windows, might need another codec if linux
        codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        out = cv2.VideoWriter("outpy_{}_{}.avi".format(data_name, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")), codec, 30, out_size)
            
        imgs = detect(yolo_cfg, data_cfg, weights, video_path, conf_thres=0.55, video=True, webcam=False)
        
        for im in imgs:
            res = cv2.resize(im, out_size)
            out.write(res)


if __name__ == "__main__":
    yolo_cfg = "cfg/yolov3-3cls.cfg"
    data_name = "full dataset.data"
    data_cfg = "data/" + data_name 
    best_weights = "weights/best.pt"
    paths = ["/home/carbor/code/yolov3/data/test_videos/indoor_test_1.MP4", "/home/carbor/code/yolov3/data/test_videos/indoor_test_2.MP4", "/home/carbor/code/yolov3/data/test_videos/volvo_test.avi"]
    img_size = 416
    resume = True # used for transfer learning
    epochs = 10000
    batch_size = 3

    training = True

    if training:
        train(
            yolo_cfg,
            data_cfg,
            img_size=img_size,
            resume=resume,
            epochs=epochs,
            batch_size=batch_size,
            multi_scale=True,
            transfer=False
        )
    
        plot_results()
        os.rename("results.txt", "results_{}.txt".format(data_name))

    for path in paths:
        test_video(path, yolo_cfg, data_cfg, best_weights, data_name)

