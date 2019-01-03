#ifndef FaceProcess_hpp
#define FaceProcess_hpp

#include <stdio.h>

#endif /* FaceProcess_hpp */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>

using namespace cv;
using namespace std;

//Class used for face collecting, training and recognizing
class FaceProcess {
public:
	cv::CascadeClassifier cascade;
	string eyedata;
public:
	FaceProcess(string cascade_path, string eyedata_path);
	int faceTrain(string fn_csv, string mode);
	int faceCollect(string collectMode, string filename, int num_each_person);
	int faceRecognition(string recognitionMode, string recog_path, string faces_txt_path, int true_label);
};

//Class used for fisherfaces predicting
class FisherPredict {
public:
	int _num_components;
	double _threshold;
	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;
	vector<Mat> _projections;
	Mat _labels;
public:
	FisherPredict(int num_components, double threshold);
	void load(const string& filename);
	int predict(InputArray _src, int true_label);
};

// Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
template<typename _Tp> void readFileNodeList(const FileNode& fn, vector<_Tp>& result);
//Normalization for image pixels
static Mat norm_0_255(InputArray _src);
//Stroing train picture and corresponding label in images&labels from filename.txt
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');
//取各个标签对应的第一张人脸图片放入到 images 中
void loadPortraits(const string& filename, vector<Mat>& images, char separator = ';');
//在frame的图像左上角顶点绘制potrait
void show_portrait(Mat &potrait, Mat &frame);
//调整图像识别尺寸
void cropFaceImage(Mat &src, Mat &dst);
