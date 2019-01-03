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
	//���ۼ������������ֶ�����


	cv::CascadeClassifier ecascade;


	/*Ԥ������������:
	����1    cropFaceRegion_percent_y :
	yֵ��������ƫ��rect.height * percent_y
	rect.y = (rect.y + rect.height * cropFaceRegion_percent_h);

	����2    cropFaceRegion_percent_h:
	percent_h height��������
	rect.height *= cropFaceRegion_percent_h;
	*/

	double cropFaceRegion_percent_y = 0.25;
	double cropFaceRegion_percent_h = 1 / 3.0;

	inline void setCropFaceRegionParams(double new_cropFaceRegion_percent_y, double new_cropFaceRegion_percent_h) {
		cropFaceRegion_percent_y = new_cropFaceRegion_percent_y;
		cropFaceRegion_percent_h = new_cropFaceRegion_percent_h;
	}

	/*���������ľ���Ϊ��׼�����г߶ȱ任
	����1:		percent_h ����ƽ�Ƹ߶�
	����2:		percent_w ����ƽ�ƿ��

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

	cv::Size des_size = cv::Size(0, 0);//δʹ��

									   /*���������ľ���Ϊ��׼�����г߶ȱ任��
									   ����1:		src			ԭͼ�����ԭͼ���иı�
									   ����2:		faceRect	������ԭͼroi Rect
									   ����3:		alignFace	���ض����face Mat
									   ����4:		draw		true ��ԭͼ�ϻ������� flase ����ԭͼ�ϻ�������
									   ����5:		scale		���ۼ��ʱͼ�����ű�
									   return		true		alignFace��Ч
									   */
	bool process(cv::Mat& src, const cv::Rect& faceRect, cv::Mat& alignFace, bool draw = true, double scale = 1.0);


	bool init(const std::string &eyedata);

private:

	void cropFaceRegion(const cv::Mat &img, const cv::Rect &facerect, cv::Mat &face, cv::Rect & rect);
	bool cropAndRotateFace(const cv::Mat image, cv::Mat& roiImg, cv::Point eye_left, cv::Point eye_right);

};

void faceAlignment();

void detectFace(const Mat &img, vector<Rect> &faces, double scale, CascadeClassifier& ffcascade);
