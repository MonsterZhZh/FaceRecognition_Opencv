#include "FaceProcess.hpp"
#include "FaceAlignment.h"

int main()
{
    FaceProcess faceProcess("/Users/monsterzhzh/Downloads/TraditionalFaceRecognition/FaceRecognition/FaceRecognition/Detection/haarcascade_frontalface_alt2.xml", "/Users/monsterzhzh/Downloads/TraditionalFaceRecognition/FaceRecognition/FaceRecognition/Detection/haarcascade_mcs_righteye.xml");
    //faceProcess.faceCollect("takephoto", "NULL", 11);
    //faceProcess.faceTrain("/Users/monsterzhzh/Downloads/TraditionalFaceRecognition/FaceRecognition/FaceRecognition/traindata/complex_photo/complex.txt", "all");
    faceProcess.faceRecognition("Fisher", "/Users/monsterzhzh/Downloads/TraditionalFaceRecognition/FaceRecognition/FaceRecognition/Recognition/MyFaceFisherModel.xml", "/Users/monsterzhzh/Downloads/TraditionalFaceRecognition/FaceRecognition/FaceRecognition/traindata/complex_photo/complex.txt", 1);
    return 0;
}


