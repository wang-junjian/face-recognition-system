from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import platform
import pickle
import random
import shutil
from datetime import datetime
from sklearn.svm import SVC
from scipy import misc
import cv2
import align.detect_face

IMAGES_ROOT_DIR = '../images/'

TRAIN_DIR = IMAGES_ROOT_DIR + 'train/'
TRAIN_MTCNN_DIR = IMAGES_ROOT_DIR + 'train_mtcnn/'

CLASSIFIER_DIR = IMAGES_ROOT_DIR + 'classifier/'
DETECT_DIR = IMAGES_ROOT_DIR + 'detect/'

CAPTURE_DIR = IMAGES_ROOT_DIR + 'capture/'
CAPTURE_REALTIME_DIR = CAPTURE_DIR + 'realtime/'

CAPTURE_MTCNN_DIR = IMAGES_ROOT_DIR + 'capture_mtcnn/'
CAPTURE_MTCNN_REALTIME_DIR = CAPTURE_MTCNN_DIR + 'realtime/'

COLLECT_FACE_TIME_S = 10

MS_MODEL = '../models/20170512-110547/'
CLASSIFIER_FILENAME = '../models/classifier.pkl'

CLASS_PROBABILITY_THRESHOLD = 0.9


def main(args):
    mode = 'CLASSIFIER'

    if not os.path.exists(CLASSIFIER_FILENAME):
        mode = 'TRAIN'

    cap = cv2.VideoCapture(0)
    print('摄像头打开', '成功' if cap.isOpened() else '失败')

    if not cap.isOpened():
        return

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=666)

            print('创建mtcnn网络和加载参数')
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
                _sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with _sess.as_default():
                    pnet, rnet, onet = align.detect_face.create_mtcnn(_sess, None)

            print('加载特征提取模型')
            facenet.load_model(MS_MODEL)

            print('按c键将进入训练模式')
            print('按空格键将进入暂停模式')
            print('按q键将退出程序')

            while True:
                key_code = cv2.waitKey(1)
                if key_code == ord('c') or mode == 'TRAIN':    # 采集样本
                    capture_train_samples(cap)

                    if os.path.exists(TRAIN_MTCNN_DIR):
                        shutil.rmtree(TRAIN_MTCNN_DIR)

                    align_dataset_mtcnn(pnet, rnet, onet, TRAIN_DIR, TRAIN_MTCNN_DIR, args)

                    mode = 'TRAIN'
                elif key_code == 32:  # 暂停程序
                    cv2.waitKey()
                elif key_code == ord('q'):  # 退出程序
                    break

                if os.path.exists(CAPTURE_REALTIME_DIR):
                    shutil.rmtree(CAPTURE_REALTIME_DIR)
                os.makedirs(CAPTURE_REALTIME_DIR)

                data_dir = TRAIN_MTCNN_DIR
                if mode == 'CLASSIFIER':
                    _, frame = cap.read()
                    frame = cv2.flip(frame, 1, 0)  # 翻转以充当镜子
                    cv2.imshow('Capture Face Sample', frame)
                    cv2.imwrite('{dir}{filename}.jpg'.format(dir=CAPTURE_REALTIME_DIR, filename=current_time_string()), frame)

                    if os.path.exists(CAPTURE_MTCNN_REALTIME_DIR):
                        shutil.rmtree(CAPTURE_MTCNN_REALTIME_DIR)

                    align_dataset_mtcnn(pnet, rnet, onet, CAPTURE_DIR, CAPTURE_MTCNN_DIR, args)

                    if not os.path.exists(CAPTURE_MTCNN_REALTIME_DIR) or not os.listdir(CAPTURE_MTCNN_REALTIME_DIR):
                        continue

                    data_dir = CAPTURE_MTCNN_DIR

                dataset = facenet.get_dataset(data_dir)

                # 检查每个类别是否至少有一个训练图像
                for cls in dataset:
                    assert (len(cls.image_paths) > 0, '数据集中每个类别必须至少有一个图像')

                paths, labels = facenet.get_image_paths_and_labels(dataset)

                print('类别数量: %d' % len(dataset))
                print('图像数量: %d' % len(paths))

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * args.batch_size
                    end_index = min((i + 1) * args.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename_exp = os.path.expanduser(CLASSIFIER_FILENAME)

                if mode == 'TRAIN':
                    # 训练分类器
                    print('训练分类器')
                    model = SVC(kernel='linear', probability=True)
                    model.fit(emb_array, labels)

                    # 创建类别名称列表
                    class_names = [cls.name.replace('_', ' ') for cls in dataset]

                    # 保存分类器模型
                    with open(classifier_filename_exp, 'wb') as outfile:
                        pickle.dump((model, class_names), outfile)
                    print('保存分类器模型到文件 "%s"' % classifier_filename_exp)

                    mode = 'CLASSIFIER'
                elif mode == 'CLASSIFIER':
                    output_dir = os.path.expanduser(CLASSIFIER_DIR)

                    # 分类图像
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)

                    print('从文件加载分类器模型 "%s"' % classifier_filename_exp)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    print('class_names', class_names, len(class_names))
                    for i in range(len(best_class_indices)):
                        class_name = class_names[best_class_indices[i]]
                        class_probability = best_class_probabilities[i]

                        if class_probability < CLASS_PROBABILITY_THRESHOLD:
                            class_name = '未知'

                        if platform.system() == 'Darwin':
                            os.system('say %s' % class_name)

                        image_file = paths[i]
                        d = '%s%s/' % (output_dir, class_name)
                        if not os.path.exists(d):
                            os.makedirs(d)
                        shutil.copyfile(image_file, '%s%s/%.3f.png' % (output_dir, class_name, class_probability))
                        print('%4d  %s: %.3f' % (i, class_name, class_probability))

                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print('准确率: %.3f' % accuracy)

    cap.release()
    cv2.destroyAllWindows()


def capture_train_samples(cap):
    """采集用于人脸分类的样本"""

    while True:
        class_name = input('注：名字为空，退出该模块\n请输入名字：')
        if not class_name:
            break

        class_name_dir = '{}{}/'.format(TRAIN_DIR, class_name)
        if not os.path.exists(class_name_dir):
            os.makedirs(class_name_dir)

        print('开始上下左右转动头......')

        begin_time = datetime.now().timestamp()
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1, 0)  # 翻转以充当镜子

            # cv2.imshow(class_name, frame) # 中文乱码未解决
            cv2.imshow('Capture Face Sample', frame)
            cv2.imwrite('{dir}{filename}.jpg'.format(dir=class_name_dir, filename=current_time_string()), frame)

            cur_time = datetime.now().timestamp()
            if cur_time - begin_time > COLLECT_FACE_TIME_S:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('类别 {} 样本数 {} 采集目录 {}'.format(
            class_name, sum([len(files) for _, _, files in os.walk(class_name_dir)]), os.path.abspath(class_name_dir)))


def current_time_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def align_dataset_mtcnn(pnet, rnet, onet, input_dir, output_dir, args, is_rectangle_face=False):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = facenet.get_dataset(input_dir)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if args.random_order:
        random.shuffle(dataset)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if args.random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('无法对齐 "%s"' % image_path)
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]

                    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            if args.detect_multiple_faces:
                                for i in range(nrof_faces):
                                    det_arr.append(np.squeeze(det[i]))
                            else:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(
                                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det_arr.append(det[index, :])
                        else:
                            det_arr.append(np.squeeze(det))

                        image = cv2.imread(image_path)
                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])

                            if not is_rectangle_face:
                                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                            else:
                                cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

                        if is_rectangle_face:
                            filename_base, file_extension = os.path.splitext(output_filename)
                            all_face_rectangle_filename = "{}{}".format(filename_base, file_extension)
                            cv2.imwrite(all_face_rectangle_filename, image)
                    else:
                        print('无法对齐 "%s"' % image_path)

    print('图像总数: %d' % nrof_images_total)
    print('成功对齐的图像数量: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
