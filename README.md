# face_detection_and_recognition_yolov5
使用yolov5构建人脸检测模型，使用预训练的Arcface完成人脸特征提取和识别

人脸数据集：http://shuoyang1213.me/WIDERFACE/


训练好的权重文件：https://pan.baidu.com/s/1XH4tFX6EH0WLVYpWtVxbFQ 提取码：j8b6
## 使用方法
使用fastapi构建了一个web接口，可以将模型部署在服务器，前端使用http协议访问。
1. 部署
   1. 修改interface_about_face_recognition.py中的 weights 变量地址为本地的yolo权重文件路径，以及第123行的arcface权重
   2. 确认本机已经配置了yolov5所必须的环境，https://github.com/ultralytics/yolov5/blob/master/requirements.txt
   3. 确认已经安装了fastapi和uvicorn两个用于构建接口的第三方库
   4. 在interface_about_face_recognition.py同级目录下新建face_img_database文件夹
   5. 运行interface_about_face_recognition.py文件即可
   6. 后台检测到人脸后会和face_img_database文件夹中的人脸特征进行对比，如果匹配到了则返回该文件夹的名称作为人脸名，否则保存该人脸到face_img_database文件夹，文件名以unkonwx.jpg递增
2. 测试
   1. test_interface文件为测试用例，使用摄像头时时捕获人脸并送去服务器检测，使用前请确保机器中装有摄像头
   2. 修改detect函数中的 post地址'192.168.0.101'为服务器所在地址，可以使用ipconfig命令查看服务器地址，如果使用同一台机器启动该项目，则地址可改为'127.0.0.1'或者'0.0.0.0'。
   3. 本机仅需opencv和requests两个环境即可
   4. 运行interface_of_model文件开始测试

3. 自己训练
   1. 请前去yolov5官网学习训练自己数据的方法
   2. 本项目中的split_oridataset.py文件可帮助你快速将WIDERFACE数据集转换为yolo的训练格式
## 最后
本项目使用yolov5s model构建，训练的速度十分快，测试准确率也很高。感兴趣的朋友可以https://github.com/ultralytics/yolov5 去官网查看更多教程，如果想对视频文件或者单帧图片进行测试，可以使用yolov5项目自带的detect文件进行测试。
人脸识别部分使用acrface网络完成，即本项目中的iresnet.py文件，该文件可对送入的人脸进行embeding嵌入，得到高维空间的特征向量，用于和数据库中的人脸信息进行比对，你可以在这里找到更详细的说明 https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch 。