import numpy as np
from PIL import Image
def show_result(result, old_img):
    result = np.reshape(result, [1, -1])
    result = result.reshape([600, 800])
    w, h = np.array(old_img).shape[0], np.array(old_img).shape[1]
    seg_img = np.zeros((600, 800, 3))
    palette = [[255, 255, 255], [174, 221, 153], [14, 205, 173], [238, 137, 39], [244, 97, 150]]
    colors = palette
    for c in range(0, 5):
        seg_img[:, :, 0] += ((result[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((result[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((result[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = Image.fromarray(np.uint8(seg_img)).resize([h, w], Image.ANTIALIAS)
    image = Image.blend(old_img, seg_img, 0.5)
    return seg_img, image