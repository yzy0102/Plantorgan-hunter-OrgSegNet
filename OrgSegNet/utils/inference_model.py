import onnxruntime as ort
import numpy as np
from PIL import Image
from utils.preprocess import preprocess
from utils.show_result import show_result
from utils.split_img import Crop_img
import warnings
warnings.filterwarnings("ignore")
#from time import ctime
def img_read(path):
    '''
    主要用于读取图像, 对图像做预处理
    返回经过处理的图像数据input_data    和   原始图像数据old_img
    :param path: 图像的存储路径
    :return: input_data, old_img
    '''
    #读入图像:
    img_ori = Image.open(path).convert("RGB")
    #割去原图的一些白边
    #img_ori = Crop_img(img_ori)
    # 暂存原图, 用于后续变换
    old_img = img_ori
    # plt.figure()
    # plt.imshow(img_ori)
    # plt.show()
    img_ori = img_ori.resize([800, 600], Image.ANTIALIAS)
    img_ori = np.array(img_ori)
    input_data = np.transpose(img_ori, (2, 0, 1))
    input_data = preprocess(input_data)
    input_data = input_data.reshape([1, 3, 600, 800])
    return input_data, old_img


def inference_model(path):
    '''
    推理模型, 得到结果
    返回值: 切割过的原始图old_img, 模型预测结果seg_img, 网页显示结果image
    :return: old_img, seg_img, image
    '''

    input_data, old_img = img_read(path)
    # 导入oonx模型, 模型推理
    #这里保存了模型的路径
    onnx_path = r"C:\Users\user\Desktop\OrgSegNet\model_path/model.onnx"
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider','CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    result = sess.run([], {input_name: input_data})
    #处理图像, image作为网页端的返回图像
    seg_img, image = show_result(result, old_img)
    return old_img, seg_img, image




if __name__ == '__main__':

    #print('***Main start***', '时间', ctime())
    path = r"E:\科研文件\Unet\jpg/A_026.jpg"
    inference_model(path)
    #print('***Main start***', '时间', ctime())
