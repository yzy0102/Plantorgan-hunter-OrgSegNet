from PIL import Image
from utils.inference_model import inference_model
import os
from scipy.stats import norm
import cv2
import numpy as np
from utils.split_img import ReturnSplitImg
from threading import Thread
from utils.intensity import get_back_intensity
import pandas as pd
from utils.calculate_shape import VisGraphOther
import warnings

warnings.filterwarnings("ignore")


def func(prop, seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel):
    prop_dict = {}
    number = []
    area = []
    intensity = []

    img = ReturnSplitImg(seg_img, prop=prop)
    img = np.array(img)
    img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_1 = cv2.threshold(img_1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((40, 40), int)
    img_1 = cv2.erode(img_1, kernel, iterations=1)

    kernel = np.ones((40, 40), int)
    img_1 = cv2.dilate(img_1, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(img_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

    cv2.drawContours(img, contours, -1, (135, 150, 255), 18)
    cv2.putText(img, "count:{}".format(len(contours)), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 8.0, (255, 0, 0), 20)


    prop_path = user_path + "/" + prop
    if os.path.exists(prop_path) == False:
        os.mkdir(prop_path)

    imgtif = Image.fromarray(img_1)
    imgtif.save(prop_path + "/" + prop + "_" + "_.tif")

    num = 1

    for ii, cc in enumerate(reversed(contours)):


        if cv2.contourArea(cc) < h * w // 5000:
            continue

        mask = np.zeros((h, w, 1), np.uint8)
        cv2.drawContours(mask, [contours[ii]], 0, 255, cv2.FILLED)

        hist = cv2.calcHist([np.array(L_old)], [0], mask[:, :, 0], [256], [0, 256])
        x = np.array(hist)
        mu = np.mean(x)
        sigma = np.std(x)
        y = norm.pdf(hist, mu, sigma)

        intensity.append(y.argmin() / background_intensity)

        (xx, yy, ww, hh) = cv2.boundingRect(cc)

        area_per_organ = round(cv2.contourArea(cc) * rule_per_pixel * rule_per_pixel, 2)

        area.append(area_per_organ)

        number.append(ii + 1)
        cv2.putText(img, str(num), (xx + ww // 4, yy + hh // 4), cv2.FONT_HERSHEY_SIMPLEX, 7, (250, 128, 114), 15)
        num += 1


    img = Image.fromarray(img)
    img.save(prop_path + "/" + prop + "_" + ".png")

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
    old_img, seg_img, image = inference_model(path)
    background_intensity = get_back_intensity(old_img)

    L_old = old_img.convert("L")
    h, w = np.array(L_old).shape
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


    user_path = r"user_2"
    if os.path.exists(user_path) == False:
        os.mkdir(user_path)



    image.save(user_path + "/image" + ".jpg")

    rule_per_pixel = round(rule / rule_length, 4)

    t1 = Thread(target=func, args=("Chloroplast", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))
    t2 = Thread(target=func, args=("Mitochondrion", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))
    t3 = Thread(target=func, args=("Vacuole", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))
    t4 = Thread(target=func, args=("Nucleus", seg_img, background_intensity, L_old, h, w, user_path, rule_per_pixel))


    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    resolution = max(h, w) // 100
    t1 = Thread(target=cal_shape, args=("Chloroplast", user_path, resolution))
    t2 = Thread(target=cal_shape, args=("Mitochondrion", user_path, resolution))
    t3 = Thread(target=cal_shape, args=("Vacuole", user_path, resolution))
    t4 = Thread(target=cal_shape, args=("Nucleus", user_path, resolution))

    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()



if __name__ == '__main__':

    path = r"test_img/C040.jpg"
    rule = 2000  # nm   1um=1000nm
    rule_length = 340  # 340 pixels
    main(path, rule, rule_length)
