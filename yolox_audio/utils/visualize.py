#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2019-2021 Intflow Inc. All rights reserved.
# --Based on YOLOX made by Megavii Inc.--

import cv2
import numpy as np
import yolox_audio.utils.boxes as B
import os

__all__ = ["vis", "vis_bbox"]

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    boxes[:,0:4] = B.xyxy2cxcywh(boxes[:,0:4])
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        cx = int(box[0])
        cy = int(box[1])
        w = int(box[2])
        h = int(box[3])
        x0 = int(cx - w/2)
        y0 = int(cy - h/2)

        ###[sx1, sy1, sx2, sy2, sx3, sy3, sx4, sy4] = B.rotate_box([cx,cy,w,h,rad])
        ###[l1_x, l1_y, l2_x, l2_y, l3_x, l3_y] = landmark

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[0], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[0]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        ##cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.circle(img,((int(cx),int(cy))), radius=6, color=color, thickness=-1)
        ##cv2.ellipse(img,((int(cx),int(cy)),(int(w),int(h)),np.rad2deg(rad)),color,2)
        ###cv2.line(img,(int(sx1),int(sy1)),(int(sx2),int(sy2)),color,2)
        ###cv2.line(img,(int(sx2),int(sy2)),(int(sx3),int(sy3)),color,2)
        ###cv2.line(img,(int(sx3),int(sy3)),(int(sx4),int(sy4)),color,2)
        ###cv2.line(img,(int(sx4),int(sy4)),(int(sx1),int(sy1)),color,2)

        #cv2.circle(img, (int(l1_x),int(l1_y)), radius=4, color=(0, 0, 255), thickness=-1)
        #cv2.circle(img, (int(l2_x),int(l2_y)), radius=4, color=(0, 255, 0), thickness=-1)
        #cv2.circle(img, (int(l3_x),int(l3_y)), radius=4, color=(0, 255, 255), thickness=-1)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def vis_bbox(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        ###text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        text = '{}'.format(class_names[cls_id])
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.929, 0.694, 0.125,
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def annot_overlay(img, dets, xyxy=True):
    category_dic={0:'M',1:'W',2:'C'} #class name
    category_color={0:(255,0,0),1:(0,255,0),2:(0,0,255)} #class color
    if len(dets[0]) == 5: #rect bbox case
        for det in dets:
            if xyxy == True:
                x1=det[0]    #x
                y1=det[1]    #y
                x2=det[2]   #width
                y2=det[3]  #height
                category_id=int(det[-1])
            else:
                cx=det[1]    #x
                cy=det[2]    #y
                w=det[3]   #width
                h=det[4]  #height
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                category_id=int(det[0])
            try:
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), category_color[category_id], 2)
                cv2.putText(img, category_dic[category_id], (int(x1),int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, category_color[category_id], 1)
            except:
                print('[False Dataset!] ', x1, y1, x2, y2, category_id)
    elif len(dets[0]) >= 6: #rotated bbox case
        if xyxy == True:
            dets[:,0:4] = B.xyxy2cxcywh(dets[:,0:4])
        for det in dets:
            if xyxy == True:
                cx=det[0]    #x
                cy=det[1]    #y
                w=det[2]   #width
                h=det[3]  #height
                rad=det[4]  #radian
                category_id=int(det[5])
                landmarks=det[6:6+2*3]
            else:
                cx=det[1]    #x
                cy=det[2]    #y
                w=det[3]   #width
                h=det[4]  #height
                category_id=int(det[0])
                rad = np.arctan2(det[5],det[6])
                landmarks=det[7:7+2*3]
            seg = B.rotate_box([cx,cy,w,h,rad])
            
            sx1=seg[0]
            sy1=seg[1]
            sx2=seg[2]
            sy2=seg[3]
            sx3=seg[4]
            sy3=seg[5]
            sx4=seg[6]
            sy4=seg[7]

            ##l1_x=landmarks[0]
            ##l1_y=landmarks[1]
            ##l2_x=landmarks[2]
            ##l2_y=landmarks[3]
            ##l3_x=landmarks[4]
            ##l3_y=landmarks[5]
            try:
                #cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), category_color[category_id], 2)
                cv2.line(img,(int(sx1),int(sy1)),(int(sx2),int(sy2)),category_color[category_id],2)
                cv2.line(img,(int(sx2),int(sy2)),(int(sx3),int(sy3)),category_color[category_id],2)
                cv2.line(img,(int(sx3),int(sy3)),(int(sx4),int(sy4)),category_color[category_id],2)
                cv2.line(img,(int(sx4),int(sy4)),(int(sx1),int(sy1)),category_color[category_id],2)

                ###cv2.circle(img, (int(l1_x),int(l1_y)), radius=4, color=(0, 0, 255), thickness=-1)
                ###cv2.circle(img, (int(l2_x),int(l2_y)), radius=4, color=(0, 255, 0), thickness=-1)
                ###cv2.circle(img, (int(l3_x),int(l3_y)), radius=4, color=(0, 255, 255), thickness=-1)

                cv2.putText(img, category_dic[category_id], (int(sx1-10),int(sy1-10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, category_color[category_id], 1)
                cv2.putText(img, "{:.2f}".format(rad), (int(cx-20),int(cy-30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, category_color[category_id], 2)
            except:
                print('[False Dataset!] ', cx, cy, w, h, rad, category_id)
    return img

def write_overlay(data, target, id, path):
    # Create target Directory
    try:
        os.mkdir(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        a=1
        #print("Directory " , path ,  " already exists")
    img = data.detach().cpu().numpy()
    dets = target.copy()
    img = img*255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(dets) > 0:
        img_overlay = annot_overlay(img, dets)
        cv2.imwrite(path + '/' + str(id)  + '.jpg', img_overlay)

def write_overlay_cv(img, target, id, path):
    # Create target Directory
    try:
        os.mkdir(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        a=1
        #print("Directory " , path ,  " already exists") 
    dets = target.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(dets) > 0:
        img_overlay = annot_overlay(img, dets)
        cv2.imwrite(path + '/' + str(id)  + '.jpg', img_overlay)

def write_overlay_preproc(img, target, id, path):
    # Create target Directory
    try:
        os.mkdir(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        a=1
        #print("Directory " , path ,  " already exists") 

    img = img.transpose((1,2,0))
    img += img.min()
    img /= img.max()
    img *= 255.0
    img = img.astype(np.uint8)
    dets = target.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if len(dets) > 0:
        img_overlay = annot_overlay(img, dets, xyxy=False)
        cv2.imwrite(path + '/' + str(id)  + '.jpg', img_overlay)