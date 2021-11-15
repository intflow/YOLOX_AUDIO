# %%
import os
import json
import tqdm
import numpy as np
import cv2
import sys
c_folder = os.path.abspath(os.path.dirname(__file__))
p_folder = os.path.abspath(os.path.dirname(c_folder))
pp_folder = os.path.abspath(os.path.dirname(p_folder))
sys.path.append(c_folder)
sys.path.append(p_folder)
sys.path.append(pp_folder)
import yolox.utils.boxes as B

root = '/data/CrowdHuman/CrowdHuman_train'
img_folder_path = os.path.join(root, 'img')
train_label_path = os.path.join(root, 'label')
train_label_merge_out = os.path.join(root, 'label_coco_bbox.json')

mode = 1   #  0:train_data,   1:validation_data

json_list = []
for (Par, Subs, Files) in os.walk(train_label_path):
    for file in Files:
        ext = os.path.splitext(file)[-1]
        if ext == '.json':
            full_path = os.path.join(Par, file)
            json_list.append(full_path)

# %%
merged_label = {}
images = []
annotations = []

def _nz(x):
    if x < 0.0:
        x = 0.0
    return x

def annot_overlay_rbbox(img, dets):
        category_dic={0:'Cow',1:'Pig'} #class name
        pose_dic={0:'Standing',1:'Sitting'} #pose name
        category_color={0:(255,0,0),1:(0,255,0)} #class color
        pose_color={0:(255,255,255),1:(0,255,255)} #pose color

        for det in dets:
            
            cx=det[0]    #x
            cy=det[1]    #y
            width=det[2]   #width
            height=det[3]  #height
            radian=det[4]  #theta
            l1_x=det[5]
            l1_y=det[6]
            l2_x=det[7]
            l2_y=det[8]
            l3_x=det[9]
            l3_y=det[10]
            
            seg = B.rotate_box([cx, cy, width, height, radian])
            
            x1=seg[0]
            y1=seg[1]
            x2=seg[2]
            y2=seg[3]
            x3=seg[4]
            y3=seg[5]
            x4=seg[6]
            y4=seg[7]

            category_id=int(det[11])
            pose_id=int(det[12])
            try:
                img=cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),category_color[category_id],2)
                img=cv2.line(img,(int(x2),int(y2)),(int(x3),int(y3)),category_color[category_id],2)
                img=cv2.line(img,(int(x3),int(y3)),(int(x4),int(y4)),category_color[category_id],2)
                img=cv2.line(img,(int(x4),int(y4)),(int(x1),int(y1)),category_color[category_id],2)
                img=cv2.circle(img, (int(l1_x),int(l1_y)), radius=2, color=(0, 0, 255), thickness=2)
                img=cv2.circle(img, (int(l2_x),int(l2_y)), radius=2, color=(0, 255, 0), thickness=2)
                img=cv2.circle(img, (int(l3_x),int(l3_y)), radius=2, color=(255, 0, 0), thickness=2)
                cv2.putText(img, category_dic[category_id], (int(cx-10),int(cy-10)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, category_color[category_id], 1)
                cv2.putText(img, pose_dic[pose_id], (int(cx-10),int(cy-20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, pose_color[pose_id], 1)
                cv2.putText(img, str(radian), (int(cx-20),int(cy-30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, category_color[category_id], 2)
            except:
                print('[False Dataset!] ', cx, cy, width, height, radian)
           
        return img

id = 0
for num1, each_file in enumerate(tqdm.tqdm(json_list)):
    # if num1 > 2:
    #     break
    with open(each_file) as f:
        json_data = json.load(f)
    
    if mode == 1:
        img = cv2.imread(os.path.join(img_folder_path, os.path.split(each_file)[-1][:-5]+'.jpg'))
        h_img, w_img, c = img.shape
        images.append({'id': num1, 'file_name': os.path.split(each_file)[-1][:-5]+'.jpg', 'height':h_img, 'width':w_img})
    else:
        images.append({'id': num1, 'file_name': os.path.split(each_file)[-1][:-5]+'.jpg'})
    
    # organize ['result']['objects']
    dets = []
    for num2, each_annos in enumerate(json_data):
        id += 1
        cx = each_annos['cx']
        cy = each_annos['cy']
        width = each_annos['width']
        height = each_annos['height']
        radian = each_annos['radian']
        l1_x = _nz(each_annos['head_x'])
        l1_y = _nz(each_annos['head_y'])
        l2_x = _nz(each_annos['neck_x'])
        l2_y = _nz(each_annos['neck_y'])
        l3_x = _nz(each_annos['hip_x'])
        l3_y = _nz(each_annos['hip_y'])

        xmin = _nz(cx - 0.5*width)
        ymin = _nz(cy - 0.5*height)

        #if width > height:
        #    tmp = width
        #    width = height
        #    height = tmp
        #    radian += 0.5*np.pi
        
        #refine rotation angle
        radian *= -1
        seg = B.rotate_box([cx, cy, width, height, radian])

        #0 ~ 2pi
        if radian > np.pi*2:
            radian -= np.pi*2

        if radian < 0:
            radian = radian

        #-pi ~ pi
        if radian >= np.pi:
            radian = -1*(2*np.pi - radian)

        #print(radian)
        ####keep radians between -05pi ~ 05pi
        if radian >= np.pi*0.5:
            radian = radian - np.pi
        elif radian <= np.pi*(-0.5):
            radian = radian  + np.pi

        #Regularize pi range from -0.25pi~0.25pi
        if radian >= 0.25*np.pi:
            radian -= 0.5*np.pi
            width = each_annos['height']
            height = each_annos['width']
        if radian <= -0.25*np.pi:
            radian += 0.5*np.pi
            width = each_annos['height']
            height = each_annos['width']

        #    radian -= 2*np.pi
        #xmin = cx - 0.5 * width
        #ymin = cy - 0.5 * height
        img_id = num1
        category_id = each_annos['class']
        anno_pos = each_annos['position']
        if anno_pos == 'Standing':
            pose_id = 0
        elif anno_pos == 'Sitting':
            pose_id = 1
        else:
            raise ValueError("anno class error '{}'".format(anno_class))


        ## Convert intflow's rbbox format into coco bbox format
        #x1,y1,x2,y2 = B.rotated2rect([cx,cy,width,height,radian], w_img, h_img)
        x1,y1,x2,y2,rad = B.rotated_reform([cx,cy,width,height,radian], w_img, h_img)
        width = x2 - x1
        height = y2 - y1

        if int(category_id) > 0:
            continue

        organized_anno = [{
            "id": int(id),
            "image_id": int(img_id),
            "category_id": int(category_id),
            "pose_id": int(pose_id),
            "bbox": [
                x1,
                y1,
                width,
                height
            ],
            "landmark": [
                l1_x,
                l1_y,
                l2_x,
                l2_y,
                l3_x,
                l3_y
            ],
            "segmentation": [
                [seg[0],
                seg[1],
                seg[2],
                seg[3],
                seg[4],
                seg[5],
                seg[6],
                seg[7]]
            ],
            "area": float(width * height),
            "iscrowd": 0
        }]
        annotations.extend(organized_anno)

        det = [cx, cy, width, height, radian, l1_x, l1_y, l2_x, l2_y, l3_x, l3_y, category_id, pose_id]
        dets.append(det)

    ## Write overlay image for debug
    ##img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ##img_overlay = annot_overlay_rbbox(img, dets)
    ##cv2.imwrite('tmp_figs2/' + str(id) + '.jpg', img_overlay)

#categories = [
#    {
#        "id": 1,
#        "name": "standing"
#    },
#    {
#        "id": 2,
#        "name": "sternallying"
#    },
#    {
#        "id": 3,
#        "name": "laterallying"
#    },
#    {
#        "id": 4,
#        "name": "mounting"
#    },
#    {
#        "id": 5,
#        "name": "sitting"
#    }
#]

categories = [
    {
        "id": 0,
        "name": "body"
    }
]

poses = [
    {
        "id": 0,
        "name": "Standing"
    },
    {
        "id": 1,
        "name": "Sitting"
    }
]

merged_label['images'] = images
merged_label['annotations'] = annotations
merged_label['categories'] = categories
merged_label['poses'] = poses
# %%
with open(train_label_merge_out, 'w') as new_f:
    json.dump(merged_label, new_f)
# %%
