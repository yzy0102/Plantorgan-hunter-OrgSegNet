import numpy as np
import cv2

# 实现功能: 获取每个细胞器的图像
def ReturnSplitImg(img, prop):
    """
    用于分割细胞器图层, 输入的图像格式可以使PIL也可以是np格式.
    :param img:
    :param prop:
    :return:
    """
    palette = [[255, 255, 255], [174, 221, 153], [14, 205, 173], [238, 137, 39], [244, 97, 150]]
    try:
        if len(img.dtype) == 0:
            pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pixels = np.array(img)


    if prop == "Chloroplast":
        #print("叶绿体图像:")
        pixels[np.all(pixels != palette[1], axis=-1)] = (255, 255, 255)
    elif prop == "Mitochondrion":
        #print("线粒体图像:")
        pixels[np.all(pixels != palette[2], axis=-1)] = (255, 255, 255)
    elif prop == "Vacuole":
        #print("液泡图像:")
        pixels[np.all(pixels != palette[3], axis=-1)] = (255, 255, 255)
    elif prop == "Nucleus":
        #print("细胞核图像:")
        pixels[np.all(pixels != palette[4], axis=-1)] = (255, 255, 255)
    elif prop == "Back":
        #print("背景图像:")
        pixels[np.all(pixels != palette[0], axis=-1)] = (255, 255, 255)

    else:
        print("请选择其中一个输入, [Back, Chloroplast, Mitochondrion, Vacuole, Nucleus]")
    return pixels


def calculate_result(result, prop):
    """
    分割图像的另一种方法(对模型推理的结果进行分割)
    :param result:
    :param prop:
    :return:
    """
    if prop == "Chloroplast":
        return np.where(result[0]==1, 1, 0)
    if prop == "Mitochondrion":
        return np.where(result[0]==2, 1, 0)
    if prop == "Vacuole":
        return np.where(result[0]==3, 1, 0)
    if prop == "Nucleus":
        return np.where(result[0]==4, 1, 0)

def Crop_img(old_img):
    """
    用于裁剪输入图像的白边.
    :param old_img:
    :return:
    """
    old_img2 = np.array(old_img)
    w,h,_ = old_img2.shape
    old_img2 = cv2.cvtColor(old_img2, cv2.COLOR_RGB2GRAY)
    ret, old_img2 = cv2.threshold(old_img2, 254, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(old_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    area = []

    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    w2 = contours[max_idx][0][0][1]
    return old_img.crop([0, 0, h, w2])
