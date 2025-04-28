import cv2
import numpy as np


def overlay_segmentation(image_path, seg_path, alpha=0.5):
    """
    Overlay a semantic segmentation result on an image.

    Parameters:
    - image_path: Path to the input image (3 channels).
    - seg_path: Path to the segmentation result (1 channel).
    - alpha: Transparency level for overlaying. Default is 0.5.

    Returns:
    - result: Combined image.
    """

    # 读入图片和语义分割结果
    image = cv2.imread(image_path)
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

    # 确保语义分割结果和图片大小一致（取图片大小为准）
    h, w, _ = image.shape
    seg = seg[:h, :w]

    # 创建一个彩色的版本，你可以根据需要调整颜色
    # 在这里，我选择了蓝色(255,0,0)来表示分割的部分
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    color_seg[seg > 0] = [0, 0, 255]

    # 使用cv2.addWeighted来叠加图片和语义分割结果
    #result = cv2.addWeighted(image, 1 - alpha, color_seg, alpha, 0)
    for c in range(3):  # 对于R,G,B三个通道
        image[:, :, c] = (1 - alpha * (seg > 0)) * image[:, :, c] + alpha * (seg > 0) * color_seg[:, :, c]

    result = image

    return result


# 试用
result = overlay_segmentation(r'E:\aafa\zypro\data\t03.jpg', r'E:\aafa\zypro\data\pinjiek2075.jpg')
cv2.imwrite(r'E:\aafa\zypro\data\resultsk2075.jpg', result)

