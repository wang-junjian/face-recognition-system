# Face Recognition System(人脸识别系统)

> 基于 [facenet](https://github.com/davidsandberg/facenet) 做的人脸识别系统

## 功能
如果模型已经存在（models/classifier.pkl），程序将自动进入实时人脸识别模式，现在设置的阈值是0.90的精确度，否则将被分类为未知。如果是mac os系统还会使用语音说出类别名。
* 按c键进入训练模式，跟着提示输入用户姓名，采集10秒的样本，开始上下左右转动头，什么也不输入将结束样本采集，接着执行训练，训练结束生成模型。
* 按空格键将进入暂停模式。
* 按q键将退出程序。

## 目录结构
```
facenet
├── images
│   ├── train           采集用于训练的样本
│   ├── train_mtcnn     裁剪人脸
│   ├── capture         实时捕捉
│   ├── capture_mtcnn   裁剪人脸
│   └── classifier      分类识别
├── models
│   ├── 20170511-185253 CASIA-WebFace
│   ├── 20170512-110547 MS-Celeb-1M
│   └── classifier.pkl  分类模型
├── src
│   ├── face_recognition_system.py
...
...
```

## 运行
```bash
python3 face_recognition_system.py
```

## facenet 的功能
### 人脸检测
```bash
python3 align/align_dataset_mtcnn.py images_align_mtcnn face_images --detect_multiple_faces True
```

### 人脸比较
```bash
python3 compare.py ../models/20170512-110547 images_compare/ap_mtcnn/1.png images_compare/ap_mtcnn/2.png images_compare/ap_mtcnn/3.png images_compare/ap_mtcnn/4.png
```

### 人脸训练和分类（在自己的数据上）
```bash
python3 classifier.py TRAIN images_classifier_mtcnn ../models/20170512-110547 ../models/classifier.pkl
python3 classifier.py CLASSIFY images_classifier_mtcnn ../models/20170512-110547 ../models/classifier.pkl
```
