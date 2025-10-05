import os
import _init_paths
import time
import argparse
import os.path as osp
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pose_estimation import get_pose_estimator
from tracker import get_tracker
from classifier import get_classifier
from utils.config import Config
from utils.video import Video
from utils.drawer import Drawer
from utils import utils

def get_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--task', choices=['pose', 'track', 'action'], default='track',
                    help='inference task for pose estimation, action recognition or tracking')

    ap.add_argument("--config", type=str,
                    default="../configs/infer_trtpose_deepsort_dnn.yaml",
                    help='all inference configs for full action recognition pipeline.')
    # inference source
    ap.add_argument('--source',
                    default='../test_data/LabRobGunTestSuccess.mp4',
                    help='input source for pose estimation, if None, it will use webcam by default')

    # save path and visualization info
    ap.add_argument('--save_folder', type=str, default='../output',
                    help='just need output folder, not filename. if None, result will not save.')
    ap.add_argument('--draw_kp_numbers', action='store_true',
                    default=False, help='draw keypoints numbers when rendering results')
    ap.add_argument('--debug_track', action='store_true',
                    default=False, help='visualize tracker algorithm with bboxes')
    return ap.parse_args()

def get_suffix(args, cfg):
    suffix = []
    suffix.append(cfg.POSE.name)
    if args.task != 'pose':
        suffix.extend([cfg.TRACKER.name, cfg.TRACKER.dataset_name, cfg.TRACKER.reid_name])
        if args.task == 'action':
            suffix.extend([cfg.CLASSIFIER.name, 'torch'])
    return suffix

def main():
     # Configs
    args = get_args()
    cfg = Config(args.config)
    pose_kwargs = cfg.POSE
    clf_kwargs = cfg.CLASSIFIER
    tracker_kwargs = cfg.TRACKER

    # Initiate video/webcam
    source = args.source if args.source else 0
    video = Video(source)

    ## Initiate trtpose, deepsort and action classifier
    pose_estimator = get_pose_estimator(**pose_kwargs)
    if args.task != 'pose':
        tracker = get_tracker(**tracker_kwargs)
        if args.task == 'action':
            action_classifier = get_classifier(**clf_kwargs)

    ## initiate drawer and text for visualization
    drawer = Drawer(draw_numbers=args.draw_kp_numbers)
    user_text = {
        'text_color': 'green',
        'add_blank': True,
        'Mode': args.task,
        # MaxDist: cfg.TRACKER.max_dist,
        # MaxIoU: cfg.TRACKER.max_iou_distance,
    }
    cnt = 0
    if os.path.exists('../output/Output_trial.txt'):
        os.remove('../output/Output_trial.txt')
    # loop over the video frames
    for bgr_frame in video:
        #if cnt%60 == 0:
            #with open("../output/Output_trial.txt","a") as f:
                #print('Label 0,0,0,0,0,0,0,0,0,0', file = f)
                #if cnt/60 != 0:
                    #print('********************************************************', file = f)
        cnt += 1
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # predict pose estimation
        start_pose = time.time()
        predictions = pose_estimator.predict(rgb_frame, get_bbox=True) # return predictions which include keypoints in trtpose order, bboxes (x,y,w,h)
        # if no keypoints, update tracker's memory and it's age
        if len(predictions) == 0 and args.task != 'pose':
            debug_img = bgr_frame
            tracker.increment_ages()
        else:
            # draw keypoints only if task is 'pose'
            if args.task != 'pose':
                # Tracking
                # start_track = time.time()
                predictions = utils.convert_to_openpose_skeletons(predictions)
                predictions, debug_img = tracker.predict(rgb_frame, predictions,
                                                                debug=args.debug_track)
                # end_track = time.time() - start_track

                # Action Recognition
                if len(predictions) > 0 and args.task == 'action':
                    predictions = action_classifier.classify(predictions)

        end_pipeline = time.time() - start_pose
        # add user's desired text on render image
        user_text.update({
            'Frame': video.frame_cnt,
            'Speed': '{:.1f}ms'.format(end_pipeline*1000),
        })

        # draw predicted results on bgr_img with frame info
        render_image = drawer.render_frame(bgr_frame, predictions, **user_text)

        if video.frame_cnt == 1 and args.save_folder:
            # initiate writer for saving rendered video.
            output_suffix = get_suffix(args, cfg)
            output_path = video.get_output_file_path(
                args.save_folder, suffix=output_suffix)
            writer = video.get_writer(render_image, output_path, fps=30)

            if args.debug_track and args.task != 'pose':
                debug_output_path = output_path[:-4] + '_debug.avi'
                debug_writer = video.get_writer(debug_img, debug_output_path)
            print(f'\n[INFO] Saving video to : {output_path}')
        # show frames
        try:
            if args.debug_track and args.task != 'pose':
                debug_writer.write(debug_img)
                #utils.show(debug_img, window='debug_tracking')
            if args.save_folder:
                writer.write(render_image)
            #utils.show(render_image, window='webcam' if isinstance(source, int) else osp.basename(source))
        except StopIteration:
            break
    if args.debug_track and args.task != 'pose':
        debug_writer.release()
    if args.save_folder and len(predictions) > 0:
        writer.release()
    video.stop()

    FramePSec = 30
    TimeWindow = 2
    NumPeople = 10
    NumKP = 18
    ofstfrm = 21240

    class CrimeFC(nn.Module):
        def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(FramePSec * TimeWindow * NumPeople * NumKP * 2, 1020)
                self.fc2 = nn.Linear(1020, 510)
                self.fc3 = nn.Linear(510, 50)
                self.fc4 = nn.Linear(50, 10, bias=False)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            return x

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    CrimeFCN = CrimeFC()
    CrimeFCN.load_state_dict(torch.load('../Classifier/CrimeClassifer_FC.pth', map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
    CrimeFCN.to(device)

    classes = ['Abuse', 'Arson', 'Assault', 'Shooting', 'Vandalism', 'Burglary', 'Fighting', 'Robbery', 'Shoplifting', 'Stealing']

    with open('../output/Output_trial.txt') as f:
        predictions = []
        vsample = np.zeros(FramePSec * TimeWindow * NumPeople * NumKP * 2)
        with torch.no_grad():
            line = f.readline()
            while line:
                temp = line.split()
                ## **********************
                if len(temp) == 1:
                    flag = True
                    output = CrimeFCN(torch.Tensor(vsample).to(device))
                    ## output is the score of each category 
                    _,prediction = torch.max(output, 0)
                    predictions.append(classes[prediction])
                    ofstps = 0
                    vsample = np.roll(vsample, -360)
                    vsample[21240:21600] = 0
                    ## nodeNumber=temp[0] xpos=temp[1] ypos=temp[2]
                if len(temp) == 3:
                    if flag:
                        flag = False
                    elif int(temp[0]) <= comp:
                        ofstps += NumKP * 2
                    vsample[int(temp[0])*2+ofstfrm+ofstps] = float(temp[1])
                    vsample[int(temp[0])*2+ofstfrm+ofstps+1] = float(temp[2])
                    comp = int(temp[0])
                line = f.readline()

    Vtarget = cv2.VideoCapture(output_path)
    fps = int(round(Vtarget.get(5)))
    width = int(Vtarget.get(3))
    height = int(Vtarget.get(4))
    totalf = int(Vtarget.get(7))
    Vwrite = cv2.VideoWriter('../output/Classified.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(width,height))

    cnt = 0
    while True:
        ret, imgtarget = Vtarget.read()
        if ret:
            if cnt <= (totalf - 30):
                txtsize = cv2.getTextSize(predictions[cnt + 29], cv2.FONT_HERSHEY_COMPLEX, 2, 2)
                imgwrite = cv2.putText(imgtarget, predictions[cnt + 29], (width-txtsize[0][0],txtsize[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            else:
                imgwrite = imgtarget
            cnt += 1
            Vwrite.write(imgwrite)
        else:
            break

    Vtarget.release()
    Vwrite.release()

if __name__ == '__main__':
    main()
