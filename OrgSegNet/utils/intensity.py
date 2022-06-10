import numpy as np
import cv2
from scipy import signal
from scipy.stats import norm
def get_back_intensity(old_img):
    '''
    读入剪裁过的原始图像, 获取背景的强度
    :param old_img:
    :return: the intensity of backgroud
    '''

    hist1 = cv2.calcHist([np.array(old_img)], [0], None, [256], [0, 256])
    #除去首尾, 避免噪声影响
    hist1 = hist1[1:254]
    bins = np.arange(hist1.shape[0] + 1)
    x = np.array(hist1)
    mu = np.mean(x)  # 计算均值
    sigma = np.std(x)
    y = norm.pdf(hist1, mu, sigma)  # 拟合一条最佳正态分布曲线y
    # 滤波等
    xxx = bins[1:]
    yyy = y.ravel()
    z1 = np.polyfit(xxx, yyy, 100)  # 用100次多项式拟合
    p1 = np.poly1d(z1)  # 多项式系数
    yvals = p1(xxx)
    num_peak_3 = signal.find_peaks(yvals, distance=10)  # distance表极大值点的距离至少大于等于10个水平单位
    def get_tensity(num_peak_3=num_peak_3, y=y):
        # 取最接近且比y.argmin()大的那一个位置
        num = [n - y.argmin() for n in num_peak_3[0]]
        num = np.array(num)
        num = np.where(num < 0, 255, num)
        return np.sort(num)[0] + y.argmin()
    return get_tensity(num_peak_3, y)




