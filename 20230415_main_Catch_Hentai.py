import os
import sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, 'nudenet'))
from nudenet import NudeClassifierLite
from nudenet import NudeDetector
from utils import plot_detect, find_Hentai_Class

def main():
    # 0. Setting ---------------------------------------------------------------------------------------------------------
    # Threshold (Classify)
    # - 값이 낮을수록, Hentai 이미지에 민감해짐
    th_cls = 0.7
    # Threshold (Detection)
    # - 값이 낮을수록, Hentai 각 클래스에 대해 민감해짐
    th_detect = 0.6

    # Detection 결과에서, 발생한 경우 Hentai로 판단할 클래스 종류
    class_to_Detect = ['EXPOSED_ANUS',          # 항문 노출
                       'EXPOSED_BUTTOCKS',      # 엉덩이 노출
                       # 'COVERED_BUTTOCKS',      # 엉덩이 비노출 (실루엣 있고, 도발적)
                       'EXPOSED_BREAST_F',      # 여성 가슴 노출
                       'COVERED_GENITALIA_F',   # 여성 성기 비노출 (실루엣 있고, 도발적)
                       'EXPOSED_GENITALIA_F',   # 여성 성기 노출
                       'EXPOSED_GENITALIA_M',   # 남성 성기 노출
                       # 'COVERED_BELLY',         # 복부 비노출 (실루엣 있고, 도발적)
                       # 'EXPOSED_BELLY',         # 복부 노출
                       ]
    # Path Input
    # 1. NSFW Sample (1~4)
    path_image = r'.\sample_image\nude_sample2.png'
    # 2. Normal Sample (1~6)
    # path_image = r'.\sample_image\normal_sample6.png'
    # ------------------------------------------------------------------------------------------------------------------

    # 1. Classifier (lite ver.) ----------------------------------------------------------------------------------------
    # initialize classifier
    classifier_lite = NudeClassifierLite()
    # Classify single image
    result_classify = classifier_lite.classify(path_image)
    # ------------------------------------------------------------------------------------------------------------------

    # 2. Detector ------------------------------------------------------------------------------------------------------
    # initialize detector
    detector = NudeDetector()
    # Detect single image
    result_detect = detector.detect(path_image)
    # Detection 결과 이미지 생성 및 저장
    plot_detect(image_path=path_image, detections=result_detect, save=True)
    # ------------------------------------------------------------------------------------------------------------------

    # 3. 결과 Decision --------------------------------------------------------------------------------------------------
    # - Classification 결과가 Threshold를 넘는가?
    if result_classify[path_image]['unsafe'] >= th_cls:
        # - Detection 결과에서, Hentai 클래스 체크하고, 최종 결과 생성
        result_fin, detected_Hentai_class = find_Hentai_Class(class_to_detect=class_to_Detect,
                                                              result_detect=result_detect, th_detect=th_detect)
    else:
        result_fin = False
    # ------------------------------------------------------------------------------------------------------------------

    # 결과 출력
    if result_fin:
        print("\nDetected Hentai Image!!! (๑•́₃•̀๑)")
    else:
        print("No detected Hentai. ʕ•ﻌ•ʔ")

    # Classification Result
    print(f"Probability Hentai : {result_classify[path_image]['unsafe']}")
    # Detection Result
    try:
        if len(detected_Hentai_class) > 0:
            for element in detected_Hentai_class:
                label, score = element
                print(f"Hentai Class: {label}, Score: {score:.2f}")
    except:
        print(f"Hentai Class: {None}, Score: {None}")

if __name__ == '__main__':
    main()