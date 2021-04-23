import argparse

from utils.datasets import *
from utils.utils import *
from char_recog import *
import math
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## HyperParameters

LICENSE_PLATE_COLOR = [0,255,0]
CHARACTER_SEG_COLOR = [255,0,0]
LICENSE_TEXT = [0,0,255]

LICENSE_IMG_SIZE = 640
CHARACTER_IMG_SIZE = 256

CHAR_LIMIT_MAX = 10
CHAR_LIMIT_MIN = 8
RISK = False

rr=0.1

## Constraining the output for indian system of cars

INDIAN_SPECIFIC = True

Indian_System = {
    10: ['c','c','n','n','c','c','n','n','n','n'],
    9: ['c','c','n','n','c','n','n','n','n'],
    8: ['c','c','n','n','n','n','n','n']
}

def center(char):
    return (char[1][0]+char[1][2])/2, (char[1][1]+char[1][3])/2

def wh(char):
    return (char[1][2]-char[1][0]), (char[1][3]-char[1][1])
def dst(x1,y1,x0=0,y0=0):
    return math.sqrt((x1-x0)**2 + (y1-y0)**2)

## Detection algorithm for License Plate using Yolo-V5
def car2lp(img, im0, model):
    t1 = torch_utils.time_synchronized()
    pred = model(img)[0]
    ## Play around with the conf thres of nms
    pred = non_max_suppression(pred, 0.3, iou_thres=0.3, fast=True, classes=None, agnostic=True)
    plates = []
    t2 = torch_utils.time_synchronized()
    im0c = im0.copy()
    for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for xmin, ymin, xmax, ymax, conf, cls in det:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    img_temp = im0c[ymin:ymax,xmin:xmax,:]
                    plot_one_box((xmin,ymin,xmax,ymax), im0, label="license plate", color=LICENSE_PLATE_COLOR, line_thickness=3)
                    plates.append(img_temp)
    return plates , (t2-t1)

## Detection algorithm for Individual Characters using Yolo-v5
def lp2char(img, im0, model):   
    t1 = torch_utils.time_synchronized()
    pred = model(img)[0]
    ## Play around with the conf thres of nms
    pred = non_max_suppression(pred, 0.1, iou_thres=0.15, fast=True, classes=None, agnostic=True)
    chars = []
    t2 = torch_utils.time_synchronized()
    im0c = im0.copy()
    for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for xmin, ymin, xmax, ymax, conf, cls in det:
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    img_temp = im0c[ymin:ymax,xmin:xmax,:]
                    plot_one_box((xmin,ymin,xmax,ymax), im0, label="character", color=CHARACTER_SEG_COLOR, line_thickness=3)
                    chars.append([img_temp, (xmin, ymin, xmax, ymax, conf)])
    #print(kmean(chars))
    chars = sorted(chars,key=lambda x: x[1][-1], reverse=True)    
    if len(chars)>=CHAR_LIMIT_MAX:
        #print("Hi")
        chars = chars[:CHAR_LIMIT_MAX]
        chars = sorted(chars, key=lambda x: ((x[1][0]+x[1][2]) + 2*(x[1][1]+x[1][3]) )) 
    elif len(chars)>CHAR_LIMIT_MIN:
        #print("yolo")
        chars = sorted(chars, key=lambda x: ((x[1][0]+x[1][2]) + 2*(x[1][1]+x[1][3]) ))
        final = [0,0,0,0,0,0,0,0]
        if RISK:
            chars = kmean(chars)
        else:
            chars = sorted(chars, key=lambda x: ((x[1][0]+x[1][2]) + 2*(x[1][1]+x[1][3]) )) 
            

    else:
        if RISK:
            chars = kmean(chars)
        else:
            chars = sorted(chars, key=lambda x: ((x[1][0]+x[1][2]) + 2*(x[1][1]+x[1][3]) )) 
    
    return [char[0] for char in chars] , (t2-t1)

def conti_array(imgi):
    img1 = letterbox(imgi, new_shape=CHARACTER_IMG_SIZE)[0]
    img1 = img1[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img1 = np.ascontiguousarray(img1)
    return img1

  
## detects Licencse plate in the whole  image
def lp_detector(device, weight_path=None):

    
    half  = device.type != 'cpu'

    model = torch.load(weight_path or "weights/last_license.pt", map_location=device)['model'].float()
    model.to(device).eval()
    if half:
        model.half()

    return model

## detects individual character in the cropped image
def char_recog_predict(img, preprocess, transform, device ):
    img = preprocess(img)
    img = transform(img).to(device)
    return img

def chr_detector(device, weight_path=None):

    half  = device.type != 'cpu'

    model = torch.load(weight_path or "weights/last_cs2.pt", map_location=device)['model'].float()
    model.to(device).eval()
    if half:
        model.half()
    return model

## encoding for the classes
encoding = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
encoding = sorted(encoding)

def classify(inp,typ = None):
    y_pred_softmax = torch.log_softmax(inp, dim = 1)
    if typ is None:
        probs , y_pred = torch.max(y_pred_softmax,dim=-1)
        return [encoding[val] for val in y_pred], [str(100*(ele.item()+1)) for ele in probs]
    else:
        ans = []
        probs = []
        for tp, val in zip(typ,y_pred_softmax):
            if tp=='c':
                ans.append(encoding[torch.max(val[10:],dim=-1)[1]+10])
                probs.append(100*(torch.max(val[10:],dim=-1)[0].item()+1))
            else :
                ans.append(encoding[torch.max(val[:10],dim=-1)[1]])
                probs.append(100*(torch.max(val[:10],dim=-1)[0].item()+1))
        return ans, [str(ele) for ele in probs]
    
    
## The whole work flow of the code
def detect(source=None, video=False,full_img=False):
    device = torch_utils.select_device("0")  ## fixing it
    half  = device.type != 'cpu'
    if full_img:
        lp_yolo = lp_detector(device)
    chr_yolo = chr_detector(device)
    net = Net().to(device)
    net.load_state_dict(torch.load("weights/character_recog_up.pt"))
    net.eval()
    if source is None:
        source = "sample_plates/test3.png"
    ext =  source[-3:].upper()
    if ext=="MOV" or ext=="MP4" or ext=="AVI":
        delay = 1
    else:
        delay = 0
    dataset = LoadImages(source, img_size= LICENSE_IMG_SIZE if full_img else CHARACTER_IMG_SIZE)
    ## Loads images as stream of images
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if full_img:
            plates, t1 = car2lp(img, im0s, lp_yolo)
        else:
            t1 = 0
            plates = [im0s]


        ## iterating on each license plate detected
        for nmp, plate in enumerate(plates):
            cv2.imshow("plate",plate)
            cv2.waitKey(delay)
            pimg = conti_array(plate)
            pimg = torch.from_numpy(pimg).to(device)
            pimg = pimg.half() if half else img.float()
            pimg /= 255.0
            if pimg.ndimension() == 3:
                pimg = pimg.unsqueeze(0)
            chars, t2 = lp2char(pimg, plate, chr_yolo) ## sort and filter
            t1+=t2

            ## Runs a check on number of characters detected to constrain the output
            if len(chars)==0:
                continue
            chars_img = torch.stack([char_recog_predict(char, preprocess, use_transform, device) for char in chars])
            if INDIAN_SPECIFIC:
                if len(chars)==10:
                    labels, probs = classify(net(chars_img),typ=Indian_System[10])
                elif len(chars)==9:
                    labels, probs = classify(net(chars_img),typ=Indian_System[9])
                elif len(chars)==8:
                    labels, probs = classify(net(chars_img),typ=Indian_System[8])
                else:
                    labels, probs = classify(net(chars_img))    
            else:
                labels, probs = classify(net(chars_img))
            print(labels)
            #print(probs)
            full_seg = np.zeros((150,20,3),np.uint8)
            #plt.figure()
            space = 255*np.ones((150,20,3),dtype=np.uint8)

            ## Iterating over each segmented Character

            for i, char in enumerate(chars):
                if True:
                    white_b = 255*np.ones((50,50,3),dtype=np.uint8)
                    white_p = 255*np.ones((50,50,3),dtype=np.uint8)
                    cha = preprocess(char,l_size=50,const_pad=False)
                    cv2.putText(white_b,labels[i], (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(white_p,probs[i], (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], thickness=1, lineType=cv2.LINE_AA)
                    
                    full_seg = np.append(full_seg, np.append( np.append(cha, white_b, axis=0), white_p, axis=0), axis=1)
                    
                    full_seg = np.append(full_seg, space,axis=1)
            full_seg = np.append(full_seg, np.zeros((150,20,3),np.uint8), axis=1)
            cv2.imshow("Segmented_"+str(nmp),full_seg)
            cv2.waitKey(delay)

        ## Ouput
        print(t1)
        cv2.putText(im0s, "%0.2f" % (1/t1)  ,  (10,30) , cv2.FONT_HERSHEY_SIMPLEX, 1, [225,0,0], thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("input image",im0s)
        cv2.waitKey(delay)
    cv2.destroyAllWindows()
            


if __name__ == '__main__':
    ## sources
    ## "../PS2/test_pics"
    ## "../PS2/test_multipleCar"
    ## "../PS2/test_fullCar"
    ## "../PS2/test_video/video.mp4"
    ## "../PS2/test_video/video1.mp4"
    INDIAN_SPECIFIC = False
    source = "../PS2/test_video/video.mp4"
    with torch.no_grad():
        detect(source, full_img=True)
