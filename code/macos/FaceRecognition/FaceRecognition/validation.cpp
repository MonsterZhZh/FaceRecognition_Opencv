#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

//Specifically 10-fold cross-validation for AT&T database
double crossvalidate(vector<Mat>& images, vector<int>& labels){
    vector<Mat> trainimages;
    vector<Mat> testimages;
    vector<int> trainlabels;
    vector<int> testlabels;
    double mean_accuracy = 0.0;
    for(int i=0;i<10;i++){
        for(int j=0;j<40;j++){
            for(int iter=10*j;iter<10*(j+1);iter++){
                if(iter == (10*j+i)){
                    testimages.push_back(images[iter]);
                    testlabels.push_back(j+1);
                }
                else{
                    trainimages.push_back(images[iter]);
                    trainlabels.push_back(j+1);
                }
            }
        }
        Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
        model->train(trainimages, trainlabels);
        int correct = 0;
        for(int test=0;test<testimages.size();test++){
            Mat testimage = testimages[test];
            int predict_label = model->predict(testimage);
            if(predict_label == testlabels[test])
                correct++;
        }
        mean_accuracy += correct/40;
    }
    
    mean_accuracy = mean_accuracy / 10;
    return mean_accuracy;
}

int main() {
    // Get the path to your CSV.
    string fn_csv = string("/Users/monsterzhzh/Downloads/orl_faces/at.txt");
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    double mean_accuracy = 0.0;
    mean_accuracy = crossvalidate(images,labels);
    return 0;
}
