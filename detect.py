import cv2
from rknn.api import RKNN
import numpy as np
import time

# rknn 模型
RKNN_HELMET = './rknn_models/helmet.rknn' 
RKNN_FACEMASK = './rknn_models/facemask.rknn' 

BOX_THRESH = 0.75 # box iou
NMS_THRESH = 0.45 # 非极大抑制
IMG_SIZE = 416 # input size

HELMET_CLASS = ('head', 'helmet') # 头盔检测类别
FACEMASK_CLASS = ('mask', 'face') # 口罩检测类别

# ------- rknn yolov5检测函数，可以弄成工具包

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_confidences[...,0] >= BOX_THRESH)


    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
              [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes, model_target):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        #print('class: {}, score: {}'.format(CLASSES[cl], score))
        #print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)


        if model_target == 'helmet':
        
            if cl == 0:
                cv2.rectangle(image, (top, left), (right, bottom), (100,27,231), 2)
                cv2.putText(image, '{0}'.format(HELMET_CLASS[cl]),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100,27,231), 2)
            if cl == 1:
                cv2.rectangle(image, (top, left), (right, bottom), (50, 205, 154), 2)
                cv2.putText(image, '{0}'.format(HELMET_CLASS[cl]),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (50, 205, 154), 2)
                
        elif model_target == 'facemask':
            if cl == 1:
                cv2.rectangle(image, (top, left), (right, bottom), (0,69,255), 2)
                cv2.putText(image, '{0}'.format(FACEMASK_CLASS[cl]),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,69,255), 2)
            if cl == 0:
                cv2.rectangle(image, (top, left), (right, bottom), (144, 238, 144), 2)
                cv2.putText(image, '{0}'.format(FACEMASK_CLASS[cl]),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (144, 238, 144), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def post_process(outputs):
    
    input0_data = outputs[0]
    input1_data = outputs[1]
    input2_data = outputs[2]

    input0_data = input0_data.reshape([3,-1]+list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3,-1]+list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3,-1]+list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
    return input_data

# --------- 函数结尾




#### 口罩 + 头盔 视频检测

rknn_helmet = RKNN()
    
# 加载rknn
print('--> Loading model')
ret_helmet = rknn_helmet.load_rknn(RKNN_HELMET)
if ret_helmet != 0:
    print('Load rknn failed!')
    exit(ret_helmet)
print('done')

# 初始化环境
print('--> Init runtime environment')
ret_helmet = rknn_helmet.init_runtime()
if ret_helmet != 0:
    print('Init runtime environment failed')
    exit(ret_helmet)
print('done')


rknn_facemask = RKNN()
    
# 加载rknn
print('--> Loading model')
ret_facemask = rknn_facemask.load_rknn(RKNN_FACEMASK)
if ret_facemask != 0:
    print('Load rknn failed!')
    exit(ret_facemask)
print('done')

# 初始化环境
print('--> Init runtime environment')
ret_facemask = rknn_facemask.init_runtime()
if ret_facemask != 0:
    print('Init runtime environment failed')
    exit(ret_facemask)
print('done')



cap = cv2.VideoCapture(0) # 读取摄像头
cap.set(3, 1280) 
cap.set(4, 720) 

while True:
    # 读取当前帧
    success, frame = cap.read()
    start = time.time()
    final = cv2.resize(frame,(IMG_SIZE, IMG_SIZE))
    img = frame.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
    
    # 头盔推理
    outputs = rknn_helmet.inference(inputs=[img])
    input_data = post_process(outputs)
    helmet_boxes, helmet_classes, helmet_scores = yolov5_post_process(input_data)
    
    
    # 口罩推理
    outputs = rknn_facemask.inference(inputs=[img])
    input_data = post_process(outputs)
    facemask_boxes, facemask_classes, facemask_scores = yolov5_post_process(input_data)
    
    
    
    # 画头盔    
    if helmet_boxes is not None:
        draw(final, helmet_boxes, helmet_scores, helmet_classes, 'helmet')
    # 画口罩
    if facemask_boxes is not None:
        draw(final, facemask_boxes, facemask_scores, facemask_classes, 'facemask')
    
    # 计算fps并画出
    end = time.time()
    seconds = end - start
    fps = 1 / seconds  # 一秒钟可以处理多少帧
    fps = "%.2f fps" % fps
    cv2.putText(final, str(fps), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,60), 2)  
    print(fps)
    
    
    # 显示最后的结果
    final = cv2.resize(final,(1280, 720))
    cv2.imshow("Image", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()