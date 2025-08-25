import cv2
import numpy as np

from app.utils.image_similarity import (
    mse,
    psnr,
    ssim_index,
    histogram_comparison
)


def calculate_similarity_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Вычисляет итоговую оценку схожести двух изображений
    на основе нескольких метрик (MSE, PSNR, SSIM, Histogram).
    
    Каждая метрика возвращает значение в процентах,
    итоговая оценка — среднее арифметическое.
    
    :param img1: Первое изображение (numpy.ndarray)
    :param img2: Второе изображение (numpy.ndarray)
    :return: Средний процент схожести (float)
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    scores = []

    mse_val = mse(img1, img2)
    mse_score = max(0.0, 100.0 - mse_val / 1000)
    scores.append(mse_score)

    psnr_score = psnr(img1, img2)
    psnr_score = min(100.0, psnr_score)
    scores.append(psnr_score)

    ssim_score = ssim_index(img1, img2)
    scores.append(ssim_score)
    
    hist_score = histogram_comparison(img1, img2)
    scores.append(hist_score)

    final_score = float(np.mean(scores))

    return final_score


def calculate_similarity_from_files(path1: str, path2: str) -> float:
    """
    Загружает изображения с диска и вычисляет среднюю схожесть.
    
    :param path1: Путь к первому изображению
    :param path2: Путь ко второму изображению
    :return: Средний процент схожести (float)
    """
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        raise ValueError("Не удалось загрузить одно или оба изображения.")

    return calculate_similarity_score(img1, img2)


def calculate_similarity_from_bytes(img_bytes1: bytes, img_bytes2: bytes) -> float:
    """
    Вычисляет схожесть между двумя изображениями, переданными в бинарном формате.
    
    :param img_bytes1: Первое изображение в формате bytes
    :param img_bytes2: Второе изображение в формате bytes
    :return: Средний процент схожести (float)
    """
    nparr1 = np.frombuffer(img_bytes1, np.uint8)
    nparr2 = np.frombuffer(img_bytes2, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Не удалось декодировать одно или оба изображения из бинарных данных.")

    return calculate_similarity_score(img1, img2)