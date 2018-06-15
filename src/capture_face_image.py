from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from datetime import datetime
import cv2

IMAGES_DIR = 'images/'
TRAIN_DIR = IMAGES_DIR + 'train/'
CLASSIFIER_DIR = IMAGES_DIR + 'classifier/'
DETECT_DIR = IMAGES_DIR + 'detect/'

COLLECT_FACE_TIME_S = 10


def main(args):
    cap = cv2.VideoCapture(0)
    print('摄像头打开', '成功' if cap.isOpened() else '失败')

    if not cap.isOpened():
        return

    while True:
        class_name = input('请输入名字：')
        if not class_name:
            break

        class_name_dir = '{}{}/'.format(TRAIN_DIR, class_name)
        if not os.path.exists(class_name_dir):
            os.makedirs(class_name_dir)

        print('开始上下左右转动头......')

        begin_time = datetime.now().timestamp()
        while True:
            filename = datetime.now().strftime('%Y%m%d %H%M%S %f')

            _, frame = cap.read()
            frame = cv2.flip(frame, 1, 0)  # 翻转以充当镜子

            # cv2.imshow(class_name, frame) # 中文乱码未解决
            cv2.imshow('Capture Face Sample', frame)
            cv2.imwrite('{dir}{filename}.jpg'.format(dir=class_name_dir, filename=filename), frame)

            cur_time = datetime.now().timestamp()
            if cur_time - begin_time > COLLECT_FACE_TIME_S:
                break

            if cv2.waitKey(1) == ord('q'):
                break

        print('类别 {} 样本数 {} 采集目录 {}'.format(
            class_name, sum([len(files) for _, _, files in os.walk(class_name_dir)]), os.path.abspath(class_name_dir)))

    cap.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
