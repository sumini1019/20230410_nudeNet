import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# 0. Setting
BATCH_SIZE = 1
BOOLEAN = True

path_image_nude = [r'C:\Users\sumin\Downloads\nude_sample1.png', r'C:\Users\sumin\Downloads\nude_sample2.jpg',
                   r'C:\Users\sumin\Downloads\nude_sample3.jpg', r'C:\Users\sumin\Downloads\nude_sample4.jpg']
path_image_normal = [r'C:\Users\sumin\Downloads\normal_sample1.jpg', r'C:\Users\sumin\Downloads\normal_sample2.jpg',
                     r'C:\Users\sumin\Downloads\normal_sample3.png', r'C:\Users\sumin\Downloads\normal_sample4.png',
                     r'C:\Users\sumin\Downloads\normal_sample5.png', r'C:\Users\sumin\Downloads\normal_sample6.png']

path_image = path_image_nude[0]

for path_image in path_image_normal:
    # 1. Classify ---------------------------------------------------------------------------------------------------
    # Import module
    from nudenet import NudeClassifier

    # initialize classifier (downloads the checkpoint file automatically the first time)
    classifier = NudeClassifier()

    # Classify single image
    result_classify = classifier.classify(path_image)

    # Classify multiple images (batch prediction)
    # batch_size is optional; defaults to 4
    result_classify_multiple_nude = classifier.classify(path_image_nude, batch_size=BATCH_SIZE)
    result_classify_multiple_normal = classifier.classify(path_image_normal, batch_size=BATCH_SIZE)

    # # Classify video
    # # batch_size is optional; defaults to 4
    # result_classifiy_video = classifier.classify_video('path_to_video', batch_size=BATCH_SIZE)
    # # Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
    # #          "preds": {frame_i: {'safe': PROBABILITY, 'unsafe': PROBABILITY}, ....}}



    # 2. Classify lite --------------------------------------------------------------------------------------------------
    # Import module
    from nudenet import NudeClassifierLite

    # initialize classifier (downloads the checkpoint file automatically the first time)
    classifier_lite = NudeClassifierLite()

    # Classify single image
    result_classify_lite = classifier_lite.classify(path_image)

    # Classify multiple images (batch prediction)
    # batch_size is optional; defaults to 4
    result_classify_lite_nude = classifier_lite.classify(path_image_nude)
    result_classify_lite_normal = classifier_lite.classify(path_image_normal)





    # 3. Detector ------------------------------------------------------------------------------------------------------
    def plot_detect(image_path, detections, fast=None):
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
        # plt.show()

        if fast:
            plt.savefig(f'Result_Detection_Fast_{os.path.splitext(os.path.basename(image_path))[0]}.png')
        else:
            plt.savefig(f'Result_Detection_{os.path.splitext(os.path.basename(image_path))[0]}.png')


    # Import module
    from nudenet import NudeDetector

    # initialize detector (downloads the checkpoint file automatically the first time)
    detector = NudeDetector() # detector = NudeDetector('base') for the "base" version of detector.

    # Detect single image
    result_detect = detector.detect(path_image)
    # fast mode is ~3x faster compared to default mode with slightly lower accuracy.
    result_detect_fast = detector.detect(path_image, mode='fast')
    # Returns [{'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...]

    plot_detect(image_path=path_image, detections=result_detect, fast=False)
    plot_detect(image_path=path_image, detections=result_detect_fast, fast=True)

    # # Detect video
    # # batch_size is optional; defaults to 2
    # # show_progress is optional; defaults to True
    # detector.detect_video('path_to_video', batch_size=BATCH_SIZE, show_progress=BOOLEAN)
    # # fast mode is ~3x faster compared to default mode with slightly lower accuracy.
    # detector.detect_video('path_to_video', batch_size=BATCH_SIZE, show_progress=BOOLEAN, mode='fast')
    # # Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
    # #          "preds": {frame_i: {'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...], ....}}




    # 결과 저장
    dict_result = {'Prob_Cls': result_classify[path_image]['unsafe'],
                   'Prob_Cls_Lite': result_classify_lite[path_image]['unsafe']}
    print(f'\n{dict_result}')