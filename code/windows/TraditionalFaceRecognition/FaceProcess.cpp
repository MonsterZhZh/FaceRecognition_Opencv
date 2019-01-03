#include "FaceProcess.h"
#include "FaceAlignment.h"

/*
Methods of functions
*/

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
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
		Mat temp;
		if (!path.empty() && !classlabel.empty()) {
			//Careful: index starting from 0
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

void loadPortraits(const string& filename, vector<Mat>& images, char separator) {
	string fn_csv = string(filename);
	std::ifstream file(fn_csv, ifstream::in); //read only
	if (!file) {
		string error_message = "找不到文件，请核对路径.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	int label(0);
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			if (atoi(classlabel.c_str()) != label) {
				Mat potrait = imread(path, 0); // 8 bit gray: CV_LOAD_IMAGE_GRAYSCALE  = 0
				images.push_back(potrait);
				label = atoi(classlabel.c_str());
			}
		}
	}
}

void show_portrait(Mat &potrait, Mat &frame) {
	int channels = potrait.channels();
	int nRows = potrait.rows;
	int nCols = potrait.cols*channels;
	uchar *p_p, *p_f;
	for (auto i = 0; i<nRows; i++) {
		//通过.ptr<>函数得到一行的指针，并用[]操作符访问某一列的像素值
		p_p = potrait.ptr<uchar>(i);
		p_f = frame.ptr<uchar>(i);
		for (auto j = 0; j<nCols; j++) {
			//三通道全部存储为单通道的灰度值
			p_f[j * 3] = p_p[j];
			p_f[j * 3 + 1] = p_p[j];
			p_f[j * 3 + 2] = p_p[j];
		}
	}
}

void cropFaceImage(Mat &src, Mat &dst) {
	resize(src, src, Size(250, 250), 1, 1, INTER_CUBIC);
	Rect myROI(15, 0, 220, 220);
	src(myROI).copyTo(dst);
}

template<typename _Tp>
void readFileNodeList(const FileNode& fn, vector<_Tp>& result) {
	if (fn.type() == FileNode::SEQ) {
		for (FileNodeIterator it = fn.begin(); it != fn.end();) {
			_Tp item;
			it >> item;
			result.push_back(item);
		}
	}
}

/*
//利用LAB颜色空间，将测试图像调整光照至数据库中同一类对应图片的第一张
Mat changeIllumination(Mat train,Mat test){
cvtColor(train, train, COLOR_GRAY2BGR);
//imshow("train", train);
//waitKey(30);
cvtColor(test, test, COLOR_GRAY2BGR);
//imshow("test", test);
//waitKey(30);
Mat temp_test;
Mat temp_train;
Scalar mean_test,var_test;
Scalar mean_train,var_train;
vector<Mat> labPlane_test;
vector<Mat> labPlane_train;
cvtColor(test, temp_test, COLOR_BGR2Lab);
cvtColor(train, temp_train, COLOR_BGR2Lab);
split(temp_test, labPlane_test);
split(temp_train, labPlane_train);
meanStdDev(labPlane_train[0], mean_train,var_train);
meanStdDev(labPlane_test[0], mean_test,var_test);
labPlane_test[0] = (labPlane_test[0] - uchar(mean_test[0]))*uchar(var_train[0]/var_test[0]) + uchar(mean_train[0]);
merge(labPlane_test,temp_test);
cvtColor(temp_test, temp_test, COLOR_Lab2BGR);
cvtColor(temp_test, temp_test, COLOR_BGR2GRAY);
//imshow("temp_test",temp_test);
//waitKey(30);
return temp_test;
}
*/

/*
Class used for face collecting, training and recognizing
*/

FaceProcess::FaceProcess(string cascade_path, string eyedata_path) {
	cascade.load(cascade_path);
	eyedata = eyedata_path;
}

int FaceProcess::faceTrain(string fn_csv, string mode) {
	if (mode.empty())
	{
		cout << "Please input second parameter!!!";
		return 0;
	}

	vector<Mat> images;
	vector<int> labels;

	try
	{
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	/*
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	*/
	//避免FisherFace无法训练的情况
	if (mode.compare("all") == 0)
	{
		Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer();
		model1->train(images, labels);
		model1->save("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\Recognition\\MyFaceFisherModel.xml");
	}
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
	model->train(images, labels);
	model->save("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\Recognition\\MyFacePCAModel.xml");

	Ptr<FaceRecognizer> model2 = createLBPHFaceRecognizer();
	model2->train(images, labels);
	model2->save("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\Recognition\\MyFaceLBPHModel.xml");

	cout << "Training Successfully!" << endl;
	return 0;
}

int FaceProcess::faceCollect(string collectMode, string filename, int num_each_person) {
	VideoCapture cap;
	Mat frame;
	int pic_num = 1;
	//入库图片予以矫正
	FaceAlignment fa;
	fa.init(eyedata);
	//选择从视频流获取图片还是拍照获取
	if (collectMode == "takephoto")
	{
		cap.open(1);
		if (!cap.isOpened())
		{
			cout << "Can not open the camera!!!" << endl;
			return -1;
		}
	}
	else if (collectMode == "video")
	{
		//尚未测试通过, To be continued
		cap.open(filename);
		if (!cap.isOpened())
		{
			cout << "Can not open the video!!!" << endl;
			return -1;
		}
	}
	else if (collectMode == "photo")
	{
		//filename: 每个人图片的文件路径，命名格式例如：1.jpg, 2.jpg, ...
		//num_each_person: 每个人的照片个数
		for (int i = 1; i <= num_each_person; i++)
		{
			Mat temp;
			std::vector<Rect> faces;
			//Mat temp_gray;
			string path_img = format("/%d.jpg", i);
			path_img = filename + path_img;
			temp = imread(path_img);
			//cvtColor(temp, temp_gray, COLOR_BGR2GRAY);
			detectFace(temp, faces, 2.0, cascade);
			if (faces.size() != 0)
			{
				Mat myFace;
				//对入库人脸进行矫正
				if (fa.process(temp, faces[0], myFace, false, 2.0)) {
					//注意这里temp会被画框
					//入库图片灰度化
					cvtColor(myFace, myFace, COLOR_BGR2GRAY);
					//入库图片尺度归一化
					Mat crop_myFace;
					cropFaceImage(myFace, crop_myFace);
					//入库图片直方图均衡化，加强对比度
					equalizeHist(crop_myFace, crop_myFace);
					string savepath = format("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\pre-process\\%d.jpg", i);
					imwrite(savepath, crop_myFace);
				}
			}
			else
				cout << "Detection Failed!!!" << endl;
		}
		return 0;
	}
	else
		cout << "Please input options: takephoto or video followed by path or photo with numbers!!!" << endl;
	while (true)
	{
		cap >> frame;
		std::vector<Rect> faces;
		//Mat frame_gray;
		//cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		size_t max_i = 0;
		detectFace(frame, faces, 2.0, cascade);
		//防止人脸矫正时在原图中绘框
		Mat temp;
		frame.copyTo(temp);
		//在获取的图像帧上绘制人脸区域
		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
		}
		//寻找最大的矩形，认为是人脸，检测函数中已设置最大目标，获得 size=1，健壮性考虑
		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0)
			{
				if (faces[i].area() > faces[max_i].area()) {
					max_i = i;
				}
			}
		}

		if (faces.size() != 0)
		{
			Mat myFace;
			//Print on upleft position of Rect area
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 255), 2);
			//入库图片人脸矫正
			if (fa.process(temp, faces[max_i], myFace, false, 2.0)) {
				//入库图片灰度化
				cvtColor(myFace, myFace, COLOR_BGR2GRAY);
				//入库图片尺度归一化
				Mat crop_myFace;
				cropFaceImage(myFace, crop_myFace);
				//入库图片直方图均衡化
				equalizeHist(crop_myFace, crop_myFace);
				string store_path = format("F:\\workspace\\TraditionalFaceRecognition\\TraditionalFaceRecognition\\pre-process\\%d.jpg", pic_num);
				imwrite(store_path, crop_myFace);
				//显示矫正图片
				imshow(store_path, crop_myFace);
				waitKey(500);
				destroyWindow(store_path);
				pic_num++;
			}
			//每个人采集12张图片，尺寸均为220 * 220
			if (pic_num == num_each_person || frame.empty())
			{
				cap.release();
				return 0;
			}
		}
		imshow("frame", frame);
		waitKey(100);
	}
	return 0;
}

int FaceProcess::faceRecognition(string recognitionMode, string recog_path, string faces_txt_path, int true_label) {
	//recog_path: 输入要载入的模型文件
	//faces_txt_path: 要载入的人脸数据库标记文件
	VideoCapture cap(0);    //打开摄像头(0 for embeded camera,1 for usb connected camera)
							//记录检测过程，存储为视频
							// double rate = 5.0;  //视频帧率
							// Size videosize = Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT)); //视频尺寸
							// VideoWriter writer("Recognition.mp4", CV_FOURCC('D', 'I', 'V', 'X'), rate, videosize); //人脸检测视频,MPEG-4 Coding

	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	//选择识别模型
	//Ptr<FaceRecognizer> model;
	FisherPredict model(1, 1800);
	//训练好的文件名称，放置在可执行文件目录下
	FaceAlignment fa;
	fa.init(eyedata);
	//人脸识别初始化
	if (recognitionMode == "LBPH")
	{
		//model  = createLBPHFaceRecognizer(1,8,8,8,100);
	}
	else if (recognitionMode == "Fisher")
	{
		//model = createFisherFaceRecognizer(1,1500);
	}
	else if (recognitionMode == "Eigen")
	{
		//model = createEigenFaceRecognizer();
	}
	else
	{
		cout << "Please choose a mode to train!!!" << endl;
		return -1;
	}
	//model->load(recog_path);
	model.load(recog_path);
	//用于在左上角显示对应标签的缩略图
	vector<Mat> potraits;
	loadPortraits(faces_txt_path, potraits);

	while (1)
	{
		double start_detect = (double)getTickCount();
		cap >> frame;

		//建立用于存放检测到人脸的向量容器
		vector<Rect> faces(0);
		detectFace(frame, faces, 2.0, cascade);
		//图像帧上输出标签与距离所在点
		Point text_lb;
		//寻找最大的矩形，认为是人脸
		size_t max_i = 0;

		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0)
			{
				if (faces[i].area() > faces[max_i].area()) {
					max_i = i;
				}
			}
		}

		//用于获取人脸检测时间
		double detect_time = (double)getTickCount();
		cout << "Detection time is:" << (detect_time - start_detect)*1000. / getTickFrequency() << "ms!" << endl;

		if (faces.size() != 0)
		{
			cout << "Detected Face!!!" << endl;
			text_lb = Point(faces[max_i].x - 40, faces[max_i].y);
			//显示检测到的人脸框
			rectangle(frame, faces[max_i], Scalar(255, 0, 0), 3, 8, 0);
			//检测结果写入到视频中
			//writer << frame;
			namedWindow("Detected Face", CV_WINDOW_NORMAL);
			imshow("Detected Face", frame);
			//人脸矫正
			Mat face_test;
			Mat temp;
			frame.copyTo(temp);
			Mat alignface;

			if (fa.process(temp, faces[max_i], alignface, true, 2.0))
			{
				cout << "Detected Eyes!!!" << endl;
				double start_recognize = (double)getTickCount();
				//待检测图片灰度化
				cvtColor(alignface, alignface, CV_BGR2GRAY);
				//待检测图片尺度归一化
				cropFaceImage(alignface, face_test);
				//resize(face_test, face_test, Size(220,220));
				//待检测图片直方图均衡化
				equalizeHist(face_test, face_test);
				//光亮矫正
				//face_test = changeIllumination(potraits[atoi(argv[4])-1], face_test);
				int predict_label = 0;
				double predict_confidence = 0.0;
				//predict_label = model->predict(face_test);
				//model->predict(face_test,predict_label,predict_confidence);
				int corrects = model.predict(face_test, true_label);
				string name;
				//单纯使用最近邻匹配是有问题的，例如训练库中没有测试对象时仍然进行匹配，如此进一步加入距离限制：设置阈值
				// Norm_L2 distance -> cosine distance?
				// face verification rather than face identification
				//参考 _threshold 距离：
				//      Fisherfaces: 1500
				//      LBPH: 60
				//if(predict_label == true_label)
				if (corrects > 5)
				{
					//name = format("predicted_Label =  %d Matching_distance = %f",predict_label,predict_confidence);
					name = format("Matched!!!");
					//show_portrait函数占用时间：4ms 左右
					show_portrait(potraits[true_label - 1], temp);
					//if(predict_label > 0)
					//show_portrait(potraits[predict_label-1], temp);
				}
				else
				{
					name = format("No Matched!!!");
				}
				putText(temp, name, text_lb, FONT_HERSHEY_COMPLEX, 0.6, Scalar(0, 0, 255), 2.0);

				namedWindow("Aligned Face", CV_WINDOW_NORMAL);
				imshow("Aligned Face", alignface);
				//识别结果写入到视频中
				// writer << temp;
				namedWindow("Aligned&Recognized Face", CV_WINDOW_NORMAL);
				imshow("Aligned&Recognized Face", temp);
				//获取人脸识别时间
				double recognize_time = (double)getTickCount();
				cout << "The recognition time is:" << (recognize_time - start_recognize)*1000. / getTickFrequency() << "ms!" << endl;
			}

		}
		if (waitKey(25) == 27)
			break;
	}
	/*
	cout << "Model Information:" << endl;
	string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
	model->getInt("radius"),
	model->getInt("neighbors"),
	model->getInt("grid_x"),
	model->getInt("grid_y"),
	model->getDouble("threshold"));
	cout << model_info << endl;
	// We could get the histograms for example:
	vector<Mat> histograms = model->getMatVector("histograms");
	// But should I really visualize it? Probably the length is interesting:
	cout << "Size of the histograms: " << histograms[0].total() << endl;
	*/
	cap.release();
	destroyAllWindows();
	return 0;
}

/*
Class used for fisherfaces predicting
*/

FisherPredict::FisherPredict(int num_components, double threshold) {
	_num_components = num_components;
	_threshold = threshold;
}

void FisherPredict::load(const string& filename) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		CV_Error(CV_StsError, "File can't be opened for writing!");
	//read matrices
	fs["num_components"] >> _num_components;
	fs["mean"] >> _mean;
	fs["eigenvalues"] >> _eigenvalues;
	fs["eigenvectors"] >> _eigenvectors;
	// read sequences
	readFileNodeList(fs["projections"], _projections);
	fs["labels"] >> _labels;
	fs.release();
}

int FisherPredict::predict(InputArray _src, int true_label) {
	int correct_counts = 0;
	Mat src = _src.getMat();
	// check data alignment just for clearer exception messages
	if (_projections.empty()) {
		// throw error if no data (or simply return -1?)
		string error_message = "This Fisherfaces model is not computed yet. Did you call Fisherfaces::train?";
		CV_Error(CV_StsBadArg, error_message);
	}
	else if (src.total() != (size_t)_eigenvectors.rows) {
		string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
		CV_Error(CV_StsBadArg, error_message);
	}
	// project into LDA subspace
	Mat q = subspaceProject(_eigenvectors, _mean, src.reshape(1, 1));
	// transfer 1-nearest neighbor to face verification
	double minDist = DBL_MAX;
	int minClass = -1;
	for (size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		double dist = norm(_projections[sampleIdx], q, NORM_L2);
		minClass = _labels.at<int>((int)sampleIdx);
		if ((dist < minDist) && (dist < _threshold) && (minClass == true_label)) {
			correct_counts++;
		}
	}
	return correct_counts;
}
