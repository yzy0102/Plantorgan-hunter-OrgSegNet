import onnxruntime as ort
import numpy as np
from PIL import Image
from utils.preprocess import preprocess
from utils.show_result import show_result
from utils.split_img import Crop_img
import warnings
warnings.filterwarnings("ignore")
def img_read(path):

    img_ori = Image.open(path).convert("RGB")
    old_img = img_ori
    img_ori = img_ori.resize([800, 600], Image.ANTIALIAS)
    img_ori = np.array(img_ori)
    input_data = np.transpose(img_ori, (2, 0, 1))
    input_data = preprocess(input_data)
    input_data = input_data.reshape([1, 3, 600, 800])
    return input_data, old_img


def inference_model(path):

    input_data, old_img = img_read(path)
    onnx_path = r"../model_path/model.onnx"
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    result = sess.run([], {input_name: input_data})
    seg_img, image = show_result(result, old_img)
    return old_img, seg_img, image


