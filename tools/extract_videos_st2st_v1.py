#!/usr/bin/env python3

import os
import cv2
import concurrent.futures

# input
folder_root = ""

label_file = "{}/something-something-v1-labels.csv".format(folder_root)
train_file = "{}/something-something-v1-train.csv".format(folder_root)
val_file = "{}/something-something-v1-validation.csv".format(folder_root)
test_file = "{}/something-something-v1-test.csv".format(folder_root)
video_folder = "{}/20bn-something-something-v1/".format(folder_root)

# output
train_img_folder = "{}/training_256/".format(folder_root)
val_img_folder = "{}/validation_256/".format(folder_root)
test_img_folder = "{}/testing_256/".format(folder_root)
train_file_list = "{}/training_256.txt".format(folder_root)
val_file_list = "{}/validation_256.txt".format(folder_root)
test_file_list = "{}/testing_256.txt".format(folder_root)

def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id

id_to_label, label_to_id = load_categories(label_file)

def load_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            video_id, label_name = line.split(";")
            label_name = label_name.strip()
            videos.append([video_id, label_name])
    return videos


def load_test_video_list(file_path):
    videos = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            videos.append([line])
    return videos


train_videos = load_video_list(train_file)
val_videos = load_video_list(val_file)
test_videos = load_test_video_list(test_file)


def resize_to_short_side(h, w, short_side=256):
    newh, neww = h, w
    if h < w:
        newh = short_side
        neww = (w / h) * newh
    else:
        neww = short_side
        newh = (h / w) * neww
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww

def video_to_images(video, basedir, targetdir, short_side=256):
    try:
        cls_id = label_to_id[video[1]]
    except:
        cls_id = -1
    filename = os.path.join(basedir, video[0])
    output_foldername = os.path.join(targetdir, video[0])
    if not os.path.exists(filename):
        print("{} is not existed.".format(filename))
        return video[0], cls_id, 0
    else:
        if not os.path.exists(output_foldername):
            os.makedirs(output_foldername)
        # get frame num
        i = 0
        while True:
            img_name = os.path.join(filename + "/{:05d}.jpg".format(i + 1))
            if os.path.exists(img_name):
                output_filename = os.path.join(output_foldername + "/{:05d}.jpg".format(i + 1))
                img = cv2.imread(img_name)
                width = img.shape[1]
                height = img.shape[0]
                newh, neww = resize_to_short_side(height, width, short_side)
                img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(output_filename, img)
                i += 1
            else:
                break

        frame_num = i
        print("Finish {}, id: {} frames: {}".format(filename, cls_id, frame_num))
        return video[0], cls_id, frame_num


def create_train_video(short_side):
    with open(train_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, train_img_folder, int(short_side))
                   for video in train_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(os.path.join(train_img_folder, video_id), frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos), flush=True)
            curr_idx += 1
    print("Completed")


def create_val_video(short_side):
    with open(val_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, val_img_folder, int(short_side))
                   for video in val_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {} {}".format(os.path.join(val_img_folder, video_id), frame_num, label_id), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


def create_test_video(short_side):
    with open(test_file_list, 'w') as f, concurrent.futures.ProcessPoolExecutor(max_workers=36) as executor:
        futures = [executor.submit(video_to_images, video, video_folder, test_img_folder, int(short_side))
                   for video in test_videos]
        total_videos = len(futures)
        curr_idx = 0
        for future in concurrent.futures.as_completed(futures):
            video_id, label_id, frame_num = future.result()
            if frame_num == 0:
                print("Something wrong: {}".format(video_id))
            else:
                print("{} 1 {}".format(os.path.join(test_img_folder, video_id), frame_num), file=f, flush=True)
            print("{}/{}".format(curr_idx, total_videos))
            curr_idx += 1
    print("Completed")


create_train_video(256)
create_val_video(256)
create_test_video(256)
