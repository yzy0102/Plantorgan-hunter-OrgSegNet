from PIL import Image
from utils.inference_model import inference_model
import os
from scipy.stats import norm
import cv2
import numpy as np
from utils.split_img import ReturnSplitImg
from threading import Thread
#from time import ctime
from utils.intensity import get_back_intensity
import pandas as pd
from utils.calculate_shape import VisGraphOther
import warnings

warnings.filterwarnings("ignore")


def func(prop, seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel):
    """
    每一个线程执行的函数
    prop_dict保存了细胞器的面积和电子密度信息
    user_path为为用户创建的文件夹, 完成所有运算后, 通过 邮件(或者网页端下载的方式) 把这个文件夹发送给用户
    """

    prop_dict = {}
    number = []
    area = []
    intensity = []

    img = ReturnSplitImg(seg_img, prop=prop)
    img = np.array(img)
    img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_1 = cv2.threshold(img_1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 腐蚀
    kernel = np.ones((40, 40), int)
    img_1 = cv2.erode(img_1, kernel, iterations=1)
    # 膨胀
    kernel = np.ones((40, 40), int)
    img_1 = cv2.dilate(img_1, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    # 标识个体
    cv2.drawContours(img, contours, -1, (135, 150, 255), 18)
    cv2.putText(img, "count:{}".format(len(contours)), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 8.0, (255, 0, 0), 20)

    # 这里在每一个用户的文件夹下, 分别创建四类细胞器的文件夹, 用于保存四类细胞器的文件信息
    prop_path = user_path + "/" + prop
    if os.path.exists(prop_path) == False:
        os.mkdir(prop_path)

    imgtif = Image.fromarray(img_1)
    imgtif.save(prop_path + "/" + prop + "_" + "_.tif")
    # 遍历每一个细胞器
    num = 1
    #for ii, cc in enumerate(contours):
    for ii, cc in enumerate(reversed(contours)):

        # 限制条件，剔除一些不想要的轮廓
        if cv2.contourArea(cc) < h * w // 5000:
            continue

        mask = np.zeros((h, w, 1), np.uint8)
        cv2.drawContours(mask, [contours[ii]], 0, 255, cv2.FILLED)
        # mask_img = np.where(mask[:, :, 0] == 255, 1, 0) * np.array(L_old)
        # # 下面这行代码可以输出白色背景的细胞器
        # mask_img = np.where(mask_img[:, :] == 0, 255, mask_img)
        # # mask_couner用来处理形状
        # mask_couner = np.where(mask_img[:, :] > 0, 255, mask_img)
        # mask_couner = Image.fromarray(mask_couner).convert("RGB")
        # mask_img_rgb = Image.fromarray(mask_img).convert("RGB")
        # mask_img_rgb.save(prop_path + "/" + prop + "_" + str(ii+1) + "_.tif" )

        hist = cv2.calcHist([np.array(L_old)], [0], mask[:, :, 0], [256], [0, 256])
        x = np.array(hist)
        mu = np.mean(x)
        sigma = np.std(x)
        y = norm.pdf(hist, mu, sigma)  # 拟合一条最佳正态分布曲线y

        intensity.append(y.argmin() / background_intensity)

        (xx, yy, ww, hh) = cv2.boundingRect(cc)
        # 保存轮廓面积
        area_per_organ = round(cv2.contourArea(cc) * rule_per_pixel * rule_per_pixel, 2)

        area.append(area_per_organ)
        # 保存序号
        number.append(ii + 1)
        cv2.putText(img, str(num), (xx + ww // 4, yy + hh // 4), cv2.FONT_HERSHEY_SIMPLEX, 7, (250, 128, 114), 15)
        num += 1

    # 保存四类细胞器分割示意图
    img = Image.fromarray(img)
    img.save(prop_path + "/" + prop + "_" + ".png")
    # 保存面积和电子密度信息
    prop_dict["Serial Number"] = number
    prop_dict["area/nm2"] = area
    prop_dict["electron-density"] = intensity
    df = pd.DataFrame(prop_dict)
    df.to_csv(prop_path + "/" + prop + "_info" + ".csv", index=None)


def cal_shape(prop, user_path, resolution):
    selectedImage = user_path + "/" + prop + "/" + prop + "__.tif"
    outputFolder = user_path + "/" + prop + "/" + "shape_info"
    if os.path.exists(outputFolder) == False:
        os.mkdir(outputFolder)
    _ = VisGraphOther(selectedImage=selectedImage, resolution=resolution, outputFolder=outputFolder, inputType="image",
                      fileList=None)


def main(path, rule, rule_length):
    """
    main函数中接收了用户上传的图像image(或是图像路径)  +   用户上传的 比例尺像素长度 和 比例尺大小(单位为nm -> 两者的比值计算了单位像素的比例尺   pixel/nm)
    main函数中需要给每一个用户/或者是用户上传的每一张图像创建一个文件夹, 路径设置为  user_path
    """

    #print('***Main start***', '时间', ctime())

    # 先进行模型推理
    old_img, seg_img, image = inference_model(path)
    background_intensity = get_back_intensity(old_img)
    # 对其他细节进行计算
    L_old = old_img.convert("L")
    h, w = np.array(L_old).shape

    # 这里可以为每一个用户创建一个文件夹, 用于保存四类细胞器的文件信息
    # 每一个文件夹保存了如下的信息
    #|-user_1
    #   |--Chol
    #       |---shape_info
    #           |---LabeledShapes.png
    #           |---ShapeResultsTable.csv
    #           |---visibilityGraphs.gpickle
    #       |---Chlo__.tif
    #       |---Chlo_info.csv
    #       |---Chlo_.png
    #   |--Mito
    #   |--Nucl
    #   |--Vacu
    #   |--image.jpg

    #设置文件保存的路径!!!!!!需要为每一张照片设置
    user_path = r"user_2"
    if os.path.exists(user_path) == False:
        os.mkdir(user_path)


    # 保存分割图-用来返回到网页端输出显示
    image.save(user_path + "/image" + ".jpg")
    #print('***网页显示***', '时间', ctime())
    # ---------------至此, 网页端的显示 已经完成, 花费5s------------------------------ #
    # ------------------------------分割线--------------------------------------------------- #

    rule_per_pixel = round(rule / rule_length, 4)
    # 开了4个线程对4类细胞器进行分别处理
    t1 = Thread(target=func, args=("Chloroplast", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))
    t2 = Thread(target=func, args=("Mitochondrion", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))
    t3 = Thread(target=func, args=("Vacuole", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))
    t4 = Thread(target=func, args=("Nucleus", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))

    # 启动线程运行
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    # 等待所有线程执行完毕
    t1.join()  # 等待线程终止
    t2.join()
    t3.join()
    t4.join()
    #print('*** 分线程 end ***', '时间', ctime())

    # ------------------------------分割线---------------------------------------------------#
    # ---------------------- 到此大概2s --------- 一共7s -----------------------------------#

    # 由于计算shape_completeness比较耗费时间, 待上面4个线程结束后, 对shape进行单独计算
    resolution = max(h, w) // 100
    t1 = Thread(target=cal_shape, args=("Chloroplast", user_path, resolution))
    t2 = Thread(target=cal_shape, args=("Mitochondrion", user_path, resolution))
    t3 = Thread(target=cal_shape, args=("Vacuole", user_path, resolution))
    t4 = Thread(target=cal_shape, args=("Nucleus", user_path, resolution))
    # 启动线程运行
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    # 等待所有线程执行完毕
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    #print('***分线程2 end***', '时间', ctime())
    # --------------到这里完成了提取shape信息------------------------------------------- #


if __name__ == '__main__':
    """
    网页端获取的三个输入---图像(path)---比例尺(rule)---比例尺像素长度(rule_length)---
    """

    path = r"test_img/C040.jpg"
    rule = 2000  # 单位nm   1um=1000nm
    rule_length = 340  # 340个像素
    main(path, rule, rule_length)
