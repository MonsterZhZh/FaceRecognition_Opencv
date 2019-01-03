# Face Recognition Based on Opencv
在windows(10)和macos(High Sierra)平台上，使用opencv(2.4.13)中的Eigenface, Fisherface, LBPH算法实现实时(摄像头)人脸检测、矫正、与识别。主要包括三个文件夹：
## FaceRecognition_Opencv
问题定义，算法描述，实现流程
## FaceRecognition_Analysis
在常用人脸数据库上分析测试各个人脸识别算法，探究分析检测，识别，预处理各部分对识别效果的影响
## code
源代码，主要包括
### FaceProcess类
人脸收集，训练和识别的类
### validation
针对不同数据库进行10折交叉验证
### FaceAlignment类
根据人眼位置对人脸进行矫正的类
### create_csv
Python脚本用于生成标记训练样本的CSV文件