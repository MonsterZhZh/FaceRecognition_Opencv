//#include "opencv/cv.h"
//#include "opencv/cxcore.h"
//#include "opencv/highgui.h"
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

static double const PI = 3.14159265;

class FaceAlignment {
public:
	//人眼检测分类器参数手动调整


	cv::CascadeClassifier ecascade;


	/*预划分人眼区域:
	参数1    cropFaceRegion_percent_y :
	y值向正方向偏移rect.height * percent_y
	rect.y = (rect.y + rect.height * cropFaceRegion_percent_h);

	参数2    cropFaceRegion_percent_h:
	percent_h height保留比例
	rect.height *= cropFaceRegion_percent_h;
	*/

	double cropFaceRegion_percent_y = 0.25;
	double cropFaceRegion_percent_h = 1 / 3.0;

	inline void setCropFaceRegionParams(double new_cropFaceRegion_percent_y, double new_cropFaceRegion_percent_h) {
		cropFaceRegion_percent_y = new_cropFaceRegion_percent_y;
		cropFaceRegion_percent_h = new_cropFaceRegion_percent_h;
	}

	/*以两眼中心距离为基准，进行尺度变换
	参数1:		percent_h 左眼平移高度
	参数2:		percent_w 左眼平移宽度

	offset_w = floor(percent_w*dist);
	offset_h = floor(percent_h*dist);
	width = (dist + 2 * offset_w);
	height = (dist + 2 * offset_h);
	crop_xy(eye_left.x - offset_w, eye_left.y - offset_h);
	*/
	double percent_h = 0.5;
	double percent_w = 0.5;

	inline void setCropAndRotateFaceParams(double new_percent_h, double new_percent_w) {
		percent_h = new_percent_h;
		percent_w = new_percent_w;
	}

	cv::Size des_size = cv::Size(0, 0);//未使用

									   /*以两眼中心距离为基准，进行尺度变换：
									   参数1:		src			原图，会对原图进行改变
									   参数2:		faceRect	脸部在原图roi Rect
									   参数3:		alignFace	返回对齐的face Mat
									   参数4:		draw		true 在原图上绘制人眼 flase 不在原图上绘制人眼
									   参数5:		scale		人眼检测时图像缩放比
									   return		true		alignFace有效
									   */
	bool process(cv::Mat& src, const cv::Rect& faceRect, cv::Mat& alignFace, bool draw = true, double scale = 1.0);


	bool init(const std::string &eyedata);

private:

	void cropFaceRegion(const cv::Mat &img, const cv::Rect &facerect, cv::Mat &face, cv::Rect & rect);
	bool cropAndRotateFace(const cv::Mat image, cv::Mat& roiImg, cv::Point eye_left, cv::Point eye_right);

};

void faceAlignment();

void detectFace(const Mat &img, vector<Rect> &faces, double scale, CascadeClassifier& ffcascade);
