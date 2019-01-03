#include "FaceProcess.h"
#include "FaceAlignment.h"

int main()
{
	FaceProcess faceProcess("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\Detection\\haarcascade_frontalface_alt2.xml", "F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\Detection\\haarcascade_mcs_righteye.xml");
	//faceProcess.faceCollect("takephoto", "NULL", 11);
	//faceProcess.faceTrain("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\traindata\\complex_photo\\complex.txt", "all");
	faceProcess.faceRecognition("Fisher", "F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\Recognition\\MyFaceFisherModel.xml", "F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\traindata\\complex_photo\\complex.txt", 1);
	return 0;
}


