#include <iostream>
#include <string>
#include <memory>
#include <sys/stat.h>
#include <assert.h>
#include <unistd.h>
#include <MNN/MNNDefine.h>
#include <MNN/MNNForwardType.h>
#include <MNN/Interpreter.hpp>
#include <opencv2/opencv.hpp>
#include "time_tools.hpp"
#include "classnames.hpp"
using namespace std;
using namespace cv;

struct timespec begin_ ,end_,time_;
struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string model_name;
};

const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };


class YOLO
{
public:
	YOLO(Net_config config);
	void detect(Mat& frame);
private:
	float* anchors;
	int num_stride;
	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	int num_class;
	
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	shared_ptr<MNN::Interpreter>  net;
    MNN::ScheduleConfig net_config;
    MNN::Session* session;
    MNN::Tensor* nhwc_Tensor;
    MNN::Tensor* inputTensor;
    float* nhwc_data;
    MNN::Tensor *tensor_output;
    MNN::Tensor tensor_output_host;
    // size_t nhwc_size;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
};

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
    this->num_class = sizeof(classnames_list)/sizeof(classnames_list[0]);
	this->class_names.assign(classnames_list, classnames_list+this->num_class);
	{
		anchors = (float*)anchors_640;
		this->num_stride = 3;
		this->inpHeight = 640;
		this->inpWidth = 640;
	}
    {
        //initialize the net config and session and allocate the memory for tensor and host
        this->net = shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(config.model_name.c_str()));
        this->net_config.numThread = 10;
        this->net_config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
        MNN::BackendConfig backendConfig;
        backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
        this->net_config.backendConfig = &backendConfig;

        this->session = this->net->createSession(this->net_config);;
        this->inputTensor = this->net->getSessionInput(this->session, nullptr);
        vector<int> dims{1, this->inpHeight, this->inpWidth, 3};
        this->nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
        this->nhwc_data = this->nhwc_Tensor->host<float>();
        string output_tensor_name = "output";
        this->tensor_output  = this->net->getSessionOutput(this->session, output_tensor_name.c_str());

        // this->nhwc_size = size_t(this->nhwc_Tensor->size());
    }
    
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

void YOLO::detect(Mat& frame){
    clock_gettime(CLOCK_MONOTONIC,&begin_);
    int newh = 0, neww = 0, padh = 0, padw = 0;
    
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
    cvtColor(dstimg, dstimg, COLOR_BGR2RGB);
    dstimg.convertTo(dstimg, CV_32FC3);
    dstimg = dstimg /255.0f;

    int INPUT_SIZE = 640;

    // wrapping input tensor, convert nhwc to nchw    
    // auto nhwc_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);
    auto nhwc_size   = this->nhwc_Tensor->size();
    //copy data to host tensor memory
    std::memcpy(this->nhwc_data, dstimg.data, nhwc_size);
    //copy tensor from host to net
    this->inputTensor->copyFromHostTensor(this->nhwc_Tensor);

    // run network
    this->net->runSession(this->session);
    //copy result from net to host
    MNN::Tensor tensor_output_host(this->tensor_output, this->tensor_output->getDimensionType());
    this->tensor_output->copyToHostTensor(&tensor_output_host);
    int num_proposal = tensor_output_host.shape()[1];
    int nout = tensor_output_host.shape()[2];
    //convert tensor to float
    float* pdata = tensor_output_host.host<float>();

    //convert the result to mat format for convenient postprocess
    Mat out(num_proposal, nout, CV_32F, (void*)pdata);
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
    int n = 0, q = 0, i = 0, j = 0, row_ind = 0;
    for (n = 0; n < 3; n++)  
	{
		const float stride = pow(2, n + 3);
		int num_grid_x = (int)ceil((this->inpWidth / stride));
		int num_grid_y = (int)ceil((this->inpHeight / stride));
		for (q = 0; q < 3; q++)    ///anchor
		{
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = pdata[4];
					if (box_score > this->objThreshold)
					{
						Mat scores = out.row(row_ind).colRange(5, nout);
						Point classIdPoint;
						double max_class_socre;
						// Get the value and location of the maximum score
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre *= box_score;
						if (max_class_socre > this->confThreshold)
						{ 
							const int class_idx = classIdPoint.x;
							float cx = pdata[0];
							float cy = pdata[1];
							float w = pdata[2];
							float h = pdata[3];
							
							int left = int((cx - padw - 0.5 * w)*ratiow);
							int top = int((cy - padh - 0.5 * h)*ratioh);

							confidences.push_back((float)max_class_socre);
							boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
							classIds.push_back(class_idx);
						}
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	cout<<confidences.size()<<endl;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	clock_gettime(CLOCK_MONOTONIC,&end_);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		float confidence = confidences[idx];
		int classId = classIds[idx];
		this->drawPred(confidence, box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classId);
	}
}



int main(int argc, char** argv){
    string model_name = "model_zoo/yolov5m.mnn";
    Net_config yolo_nets = { 0.3, 0.5, 0.3,model_name};
	YOLO yolo_model(yolo_nets);
	assert(argc>1 && "Please input your images");
	if(access("demo/", F_OK)){
		cout << "Cannot find out_dir So mkdir demo..." << endl; 
		mkdir("demo/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}
    string string_img_name;
	string img_name;
    
	while(*(++argv)){
		string_img_name = *argv;
	        cout << "Predicting " + string_img_name + "..." << endl;
		Mat srcimg = imread(string_img_name);
		yolo_model.detect(srcimg);
		timespec_sub(&end_, &begin_, &time_);
		cout << "Inference time is:\n" << time_.tv_sec<<" s  "<<time_.tv_nsec/1e6<<" ms"<<endl;
		std::size_t found = string_img_name.rfind("/");
		if(found != string::npos)
			img_name = string_img_name.substr(found+1);
		else
			img_name = string_img_name;
		imwrite("demo/" + img_name, srcimg);
	}



}