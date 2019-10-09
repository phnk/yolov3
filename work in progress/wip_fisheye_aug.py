import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
import cv2
import numpy as np
from utils.datasets import *
from utils.torch_utils import *
from utils.utils import plot_images

hyp = {'giou': 1.582,  # giou loss gain
       'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou training threshold
       'lr0': 0.001,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.0,  # SGD momentum
       'weight_decay': 0.0005,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)
device = select_device()

def plot_image(img, targets, points):
    plt.imshow(img)
    img_width, img_height = img.size

    for target in targets:
        w = target[3] * img_width
        h = target[4] * img_height
        x = target[1] * img_width - w / 2
        y = target[2] * img_height - h / 2
        plt.gca().add_patch(Rectangle((x, y), w, h, edgecolor="r", facecolor="none"))
    
    plt.show()

def get_new_corner_coords(targets, width, height, camera_matrix, distortion):
    old_target_matrix = []
    
    for target in targets:
       
        class_num = target[0]
        top_left = ((target[1] - target[3] / 2) * width, (target[2] - target[4] / 2) * height)
        bottom_left = ((target[1] - target[3] / 2) * width, (target[2] + target[4] / 2) * height)
        top_right = ((target[1] + target[3] / 2) * width, (target[2] - target[4] / 2) * height)
        bottom_right = ((target[1] + target[3] / 2) * width, (target[2] + target[4] / 2) * height)

        old_target_matrix.append([class_num, top_left, bottom_left, top_right, bottom_right])

    new_bbox_coords_matrix = []
    for old_targets in old_target_matrix:
        new_bbox_coords_list = []
        current_class = old_targets[0]
        target_matrix = np.array(old_targets[1:])
        target_matrix = np.float32(target_matrix[:, np.newaxis, :])
        distorted_points = cv2.undistortPoints(target_matrix, camera_matrix, distortion, P=camera_matrix)
        
        new_bbox_coords_list.append(current_class)
        for points in distorted_points:
            new_xy = (points[0][0]), (points[0][1])
            new_bbox_coords_list.append(new_xy)
        new_bbox_coords_matrix.append(new_bbox_coords_list)

    return new_bbox_coords_matrix

def get_new_targets(new_bbox_coords_matrix, width, height):
    new_targets = []
    for bbox in new_bbox_coords_matrix:
        class_num = bbox[0]
        (top_left_x, top_left_y) = bbox[1]
        (bottom_left_x, bottom_left_y) = bbox[2]
        (top_right_x, top_right_y) = bbox[3]
        (bottom_right_x, bottom_right_y) = bbox[4]
        
        x_list = [top_left_x, bottom_left_x, top_right_x, bottom_right_x]
        y_list = [top_left_y, bottom_left_y, top_right_y, bottom_right_y]

        min_x = min(x_list)
        min_y = min(y_list)

        max_x = max(x_list)
        max_y = max(y_list)

        new_w = ((max_x - min_x) / 2) / width 
        new_h = ((max_y - min_y) / 2) / height
        (new_x, new_y) = ( (min_x / width) + new_w, (min_y / height) + new_h)

        new_targets.append([class_num, new_x, new_y, new_w * 2, new_h * 2])
    
    return new_targets

def fisheye_augmentation(img, targets=(), dx=1, dy=1, k=1, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=None):
    width, height = img.size

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    cx = width * 0.5 + dx
    cy = height * 0.5 + dy

    fx = width
    fy = width

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    distortion = np.array([k, k, 0, 0, 0], dtype=np.float32)

    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, None, None, (width, height), cv2.CV_32FC1)
    img = cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=(128, 128, 128))
    
    # get all distorted coords the image
    new_bbox_coords_matrix = get_new_corner_coords(targets, width, height, camera_matrix, distortion)

    # calculate the new targets
    new_targets = get_new_targets(new_bbox_coords_matrix, width, height)

    return img, new_targets, new_bbox_coords_matrix


if __name__  == "__main__":
    dataset = LoadImagesAndLabels("data/3 class ground and light/train/train_paths.txt", 640, augment=True)
    dataloader = torch.utils.data.DataLoader(dataset, 1, collate_fn=dataset.collate_fn)
    for imgs, targets, img_path, res in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        plot_images(imgs=imgs, targets=targets)

    
    #for i in range(10):
    #    im = Image.open("data/inside/train/images/image{}.png".format(i))
    #    with open('data/inside/train/labels/image{}.txt'.format(i), 'r') as f:
    #        targets = [[float(num) for num in line.split(' ')] for line in f]
        
    #    im, targets, new_bbox_coords_matrix = fisheye_augmentation(im, targets)
    #    im = Image.fromarray(im)

    #    plot_image(im, targets, new_bbox_coords_matrix)
