import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

abs_path = os.path.dirname(os.path.abspath(__file__))

# 2023.04.15
# - Detection 결과 시각화 목적 함수
def plot_detect(image_path, detections, save=False):
    # 원본 이미지를 연다.
    image = Image.open(image_path)

    # 이미지를 그리기 위한 객체를 만든다.
    draw = ImageDraw.Draw(image)

    # 라벨의 글씨 크기를 변경한다. (여기서는 24로 설정)
    font = ImageFont.truetype("arial.ttf", 24)

    # 각 검출 결과에 대해 바운딩 박스를 그린다.
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        score = detection['score']

        # 바운딩 박스를 그린다. (빨간색 선, 두께 3)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # 라벨과 확률을 표시한다.
        text = f"{label} {score:.2f}"
        draw.text((x1, y1 - 30), text, fill='red', font=font)

    # 결과 이미지를 출력한다.
    plt.figure(figsize=(15, 15))
    plt.imshow(np.array(image))
    plt.axis('off')

    if save:
        path_output = os.path.join(abs_path, 'output_image',
                                   f'Result_Detection_{os.path.splitext(os.path.basename(image_path))[0]}.png')
        plt.savefig(path_output)


# 2023.04.15
# - Detection 결과를 기반으로, 문제 시 되는 클래스가 있는지 확인 및 결과 반환하는 함수
def find_Hentai_Class(class_to_detect, result_detect, th_detect=0.5):
    detected_elements = []

    for item in result_detect:
        if item['label'] in class_to_detect and item['score'] > th_detect:
            detected_elements.append((item['label'], item['score']))

    if detected_elements:
        return True, detected_elements
    else:
        return False, []