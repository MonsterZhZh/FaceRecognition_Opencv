#include "FaceAlignment.h"

//static const string eyedata = "haarcascade_mcs_righteye.xml";//"data/haarcascade_mcs_righteye.xml";
//static const string mouthdata = "data/haarcascade_mcs_mouth.xml";
//static const string frontfacedata = "haarcascade_frontalface_alt2.xml";


bool FaceAlignment::init(const string &eyedata) {
	//读取人眼分类器
	if (ecascade.load(eyedata)) {
		cout << "Load eyedata completed" << endl;
		return true;
	}
	else {
		cerr << "Load eyedata error" << endl;
		return false;
	}
}


void drawRect(Mat & img, Rect & rect, Scalar color = CV_RGB(0, 0, 255)) {

	Point p1, p2;
	p1.x = cvRound(rect.x);
	p1.y = cvRound(rect.y);
	p2.x = cvRound((rect.x + rect.width - 1));
	p2.y = cvRound((rect.y + rect.height - 1));
	rectangle(img, p1, p2, color, 2);
}

void FaceAlignment::cropFaceRegion(const Mat &img, const Rect &facerect, Mat &face, Rect & rect) {
	rect = facerect;
	rect.x = rect.x;
	rect.y = static_cast<int>(rect.y + rect.height * cropFaceRegion_percent_y);//1/4还是3/10较难选择;
	rect.height = static_cast<int>(rect.height*cropFaceRegion_percent_h);//(rect.height-1)*scale/2; //只取脸上半，眉毛干扰也较大，可修改
	face = img(rect);//人脸上人眼区域预划分

					 //嘴区域预划分
					 //Rect rect2 = facerect;
					 //rect2.x = rect2.x*scale;
					 //rect2.y = (rect2.y + rect2.height / 2)*scale;
					 //rect2.width = (rect2.width - 1)*scale;
					 //rect2.height = (rect2.height - 1)*scale / 2;//(rect.height-1)*scale/2; //只取脸上半，眉毛干扰也较大，可修改

					 //Mat mouth = img(rect2);
}

void processAligment(Mat& img, vector<Rect> &eyes) {
	//对每个人眼的结果进行处理
	vector<Rect>::const_iterator r = eyes.begin();


	//mouthdetect(mouth,2);//检测嘴
}

void detectEyes(const Mat &face, vector<Rect> &eyes, CascadeClassifier& ecascade, const double & scale) {
	//face   传入图像
	//eyes   输出。在face尺度空间
	//scale  缩放比例
	//ecascade  分类器

	vector<Rect> filter;
	Mat tempg;
	Mat gray(cvRound(face.rows / scale), cvRound(face.cols / scale), CV_8UC1);
	cvtColor(face, tempg, CV_BGR2GRAY);
	resize(tempg, gray, gray.size(), 0, 0, INTER_LINEAR);//缩放，线性插值最快
	equalizeHist(gray, gray);//直方图均衡化
							 //imshow("gray", gray);


							 //人眼检测
	ecascade.detectMultiScale(gray, filter,
		1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		//|CV_HAAR_SCALE_IMAGE
		, Size(gray.cols / 4, gray.rows / 2));//检测

											  //过滤掉重叠结果
	size_t i, j;

	for (i = 0; i < filter.size(); i++) {//rect & rect 交集
		Rect r = filter[i];
		for (j = 0; j < filter.size(); j++)
			if (j != i && ((r&filter[j]).width>0 || (r&filter[j]).height>0) && (r.width + r.height)>(filter[j].width + filter[j].height))
				break;
		if (j == filter.size())
			eyes.push_back(r);
	}

	for (size_t i = 0; i < eyes.size(); ++i) {
		eyes[i].x = static_cast<int>(eyes[i].x*scale);
		eyes[i].y = static_cast<int>(eyes[i].y*scale);
		//eyes[i].width -= 1;
		eyes[i].width = static_cast<int>(eyes[i].width*scale);
		//eyes[i].height -= 1;
		eyes[i].height = static_cast<int>(eyes[i].height*scale);
	}


}


void detectFace(const Mat &img, vector<Rect> &faces, double scale, CascadeClassifier& ffcascade)
{
	//img    传入图像
	//faces  输出。在img尺度空间
	//scale  缩放比例
	//ffcascade  分类器

	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//缩放
	equalizeHist(smallImg, smallImg);

	//imshow("gray", smallImg);

	ffcascade.detectMultiScale(smallImg, faces,
		1.1, 2, 0
		| CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE
		, Size(smallImg.cols / 6, smallImg.rows / 4));//1.1, 2, 0|CV_HAAR_FIND_BIGGEST_OBJECT//|CV_HAAR_DO_ROUGH_SEARCH|CV_HAAR_SCALE_IMAGE,Size(20, 20));调整参数以加快速度

	for (size_t i = 0; i < faces.size(); ++i) {
		faces[i].x *= scale;
		faces[i].y *= scale;
		//faces[i].width -= 1;
		faces[i].width *= scale;
		//faces[i].height -= 1;
		faces[i].height *= scale;
	}
}



Mat ScaleRotateTranslate(const Mat &image, double angle, cv::Point center, double scale = 1.0) {//不改变img，返回新的mat
	int x = center.x;
	int y = center.y;
	Mat rotateMat;
	rotateMat = getRotationMatrix2D(center, angle, scale);
	Mat rotateImg;
	cv::warpAffine(image, rotateImg, rotateMat, image.size());
	return rotateImg;

}



double eyedistance(cv::Point & p1, cv::Point &p2) {
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt(dx*dx + dy*dy);
}


bool FaceAlignment::cropAndRotateFace(const cv::Mat image, Mat& roiImg, cv::Point eye_left, cv::Point eye_right) {
	//img不改变
	//img 要进行旋转的区域
	//roiImg 返回值
	//eye_left ,eye_right 左右眼中心
	//percent_h 左眼平移高度 percent_w 左眼平移宽度
	//des_size 目标尺寸
	cv::Point eye_direction(eye_right.x - eye_left.x, eye_right.y - eye_left.y);
	double rotation = 0;
	if (eye_direction.x == 0) {
		if (eye_direction.y > 0) rotation = 90;
		else rotation = -90;
	}
	else if (eye_direction.y == 0) rotation = 0;
	else rotation = atan2(double(eye_direction.y), double(eye_direction.x)) * 180 / PI;



	//以两眼中心距离为基准，进行尺度变换
	double dist = eyedistance(eye_left, eye_right);
	double offset_w = floor(percent_w*dist);
	double offset_h = floor(percent_h*dist);
	int width = static_cast<int>(dist + 2 * offset_w);
	int height = static_cast<int>(dist + 2 * offset_h);
	//double horizatal = dist + 2 * offset_h;
	//double  reference = double(des_size.width) - 2.0*offset_h;
	//double scale = dist / reference;
	Mat newImage = ScaleRotateTranslate(image, rotation, eye_left);
	cv::Point crop_xy(int(eye_left.x - offset_w), int(eye_left.y - offset_h));//重点
																			  //cv::Size crop_size(des_size.width, des_size.height);
	cv::Rect ROI(crop_xy.x, crop_xy.y, width, height);
	if (0 <= ROI.x && 0 <= ROI.width && ROI.x + ROI.width <= newImage.cols && 0 <= ROI.y && 0 <= ROI.height && ROI.y + ROI.height <= newImage.rows) {
		roiImg = newImage(ROI);
		return true;
	}
	else
		return false;
	//cv::imshow("rotate", newImage);
}

bool FaceAlignment::process(cv::Mat &src, const Rect& faceRect, Mat &alignFace, bool draw, double scale) {

	double start_align = (double)getTickCount();
	bool croped = false;

	//src会改变  if draw==true
	cv::Mat temp;
	src.copyTo(temp);
	Rect facerect = faceRect;
	Mat face = src(facerect);//进行裁剪后的face 原尺度
	Rect croprect = facerect;
	cropFaceRegion(src, facerect, face, croprect);
	vector<Rect> eyes;//face 原尺度
	detectEyes(face, eyes, ecascade, scale);
	//drawRect(src, facerect, CV_RGB(0, 0, 255));
	if (draw) {
		drawRect(src, croprect, CV_RGB(0, 255, 255));
		for (size_t j = 0; j < eyes.size(); ++j) {
			drawRect(face, eyes[j], CV_RGB(255, 0, 255));
		}
	}
	if (eyes.size() != 2)
	{
		cerr << "eye detection result is not 2 : " << eyes.size() << endl;
		return false;
		//continue;
	}
	else {
		//Mat image, cv::Point eye_left, cv::Point eye_right, double percent_h, double percent_w, cv::Size des_size
		cv::Point eye_left(int(eyes[0].x + 0.5*eyes[0].width + croprect.x), int(eyes[0].y + 0.5*eyes[0].height + croprect.y));
		cv::Point eye_right(int(eyes[1].x + 0.5*eyes[1].width + croprect.x), int(eyes[1].y + 0.5*eyes[1].height + croprect.y));

		if (eye_left.x > eye_right.x) swap(eye_left, eye_right);
		Mat newface = temp(faceRect);
		croped = cropAndRotateFace(temp, alignFace, eye_left, eye_right);
	}

	double align_time = (double)getTickCount();
	cout << "Aligning time is:" << (align_time - start_align)*1000. / getTickFrequency() << "ms!" << endl;
	return croped;
}
