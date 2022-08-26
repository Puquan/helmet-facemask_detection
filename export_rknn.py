import os
import urllib
import traceback
import time
import sys
from rknn.api import RKNN


ONNX_MODEL = './onnx_models/helmet_416_yolov5n.onnx'
RKNN_MODEL = 'helmet.rknn'
DATASET = 'dataset.txt'
QUANTIZE_ON = True
IMG_SIZE = 416 


if __name__ == '__main__':


    # 创建rknn实例
    rknn = RKNN()

    if not os.path.exists(ONNX_MODEL):
        print('model not exist')
        exit(-1)
    
    # 预处理设置
    print('--> Configuring model')
    rknn.config(batch_size=32,
                reorder_channel='0 1 2',
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3,
                target_platform = 'rk3399pro',
                output_optimize=1,
                quantize_input_node=QUANTIZE_ON)
    print('Configuration is done')

    # 加载onnx模型
    print('--> Loading model')
    
    ret = rknn.load_onnx(model=ONNX_MODEL,outputs=['onnx::Reshape_326', 'onnx::Reshape_364', 'onnx::Reshape_402'])
    
    # 加载失败则退出
    if ret != 0:
        print('Load yolov5 failed!')
        exit(ret)
    print('Loading is done')

    # 构建模型
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build yolov5 failed!')
        exit(ret)
    print('Building is done')
    


    # 导出模型
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export yolov5rknn failed!')
        exit(ret)
    print('Export is done')

