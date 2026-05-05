import cv2
import numpy as np


def preprocess_handwriting(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Не удалось загрузить изображение")
        return

    # Уменьшаем для скорости (опционально)
    scale = 2
    # scale = 0.75
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    # Адаптивная бинаризация (можно без CLAHE)
    # binary = cv2.adaptiveThreshold(img, 255,
    #                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 15, 2)
    binary = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 6)

    # Медианный фильтр
    denoised = cv2.medianBlur(binary, 3)

    # Морфологическое открытие (удаление мелкого шума)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imwrite(output_path, cleaned)
    print(f"Обработанное изображение сохранено: {output_path}")


preprocess_handwriting("texts/pngs/original/my_page_0.jpeg",
                       "texts/pngs/black_white/my_page_0.jpeg")
