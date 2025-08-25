import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Вычисляет среднеквадратичную ошибку (Mean Squared Error) между двумя изображениями.
    Чем меньше значение, тем изображения более похожи.
    
    :param img1: Первое изображение (numpy.ndarray)
    :param img2: Второе изображение (numpy.ndarray)
    :return: Значение ошибки (float)
    """
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
    return float(err)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Вычисляет отношение сигнал/шум по пику (Peak Signal-to-Noise Ratio).
    Чем выше значение, тем изображения более похожи.
    
    :param img1: Первое изображение (numpy.ndarray)
    :param img2: Второе изображение (numpy.ndarray)
    :return: Значение PSNR в децибелах (float)
    """
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 100.0
    pixel_max = 255.0
    return float(20 * math.log10(pixel_max / math.sqrt(mse_val)))


def ssim_index(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Вычисляет индекс структурного сходства (Structural Similarity Index).
    Значение от 0 до 100 (%), где 100 — изображения идентичны.
    
    :param img1: Первое изображение (numpy.ndarray)
    :param img2: Второе изображение (numpy.ndarray)
    :return: Значение схожести в процентах (float)
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return float(score * 100)


def histogram_comparison(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Сравнивает два изображения по гистограммам цветов.
    Возвращает схожесть в процентах на основе корреляции гистограмм.
    
    :param img1: Первое изображение (numpy.ndarray)
    :param img2: Второе изображение (numpy.ndarray)
    :return: Значение схожести в процентах (float)
    """
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8],
                         [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return float(score * 100)
