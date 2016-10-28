#include <string>
#include <stdlib.h>
#include <jni.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include "com_opencv_demo_OpenCVHelper.h"

#include <android/log.h>

using namespace cv;

#define   LOGI(...)  __android_log_print(ANDROID_LOG_INFO,"opencv_jni",__VA_ARGS__)

Mat getRplane(const Mat &in)
{
     vector<Mat> splitBGR(in.channels()); //容器大小为通道数3
     split(in,splitBGR);
     LOGI("-------------------split : %d",in.cols);
     LOGI("-------------------split : %d",in.rows);
     if(in.cols >600)
     {
         Mat resizeR( 450,600 , CV_8UC1);
         cv::resize( splitBGR[0] ,resizeR ,resizeR.size());
         return resizeR;
     }
     else{
       return splitBGR[0];
      }

}

void OstuBeresenThreshold(const Mat &in, Mat &out) //输入为单通道
{
    threshold(in ,out, 0,255 ,CV_THRESH_OTSU); //otsu获得全局阈值
}

bool isEligible(const RotatedRect &candidate)
{
    const float aspect = 15; //长宽比
    int min = 15*15; //最小区域
    int max = 30*aspect*30;  //最大区域

    int area = candidate.size.height * candidate.size.width;

    if(area < min || area > max) //满足该条件才认为该candidate为号码区域
        return false;
    else
        return true;
}

void posDetect(const Mat &in, vector<RotatedRect> & rects)
{


    Mat threshold_R;
    OstuBeresenThreshold(in ,threshold_R ); //二值化

    cv::imwrite("/storage/emulated/0/opencv_img/OstuBeresenThreshold.png", threshold_R);

    Mat imgInv(in.size(),in.type(),cv::Scalar(255));
    Mat threshold_Inv = imgInv - threshold_R; //黑白色反转，即背景为黑色

    cv::imwrite("/storage/emulated/0/opencv_img/threshold_R.png", threshold_Inv);

    Mat elm = getStructuringElement(MORPH_RECT ,Size(2 ,2));
    erode(threshold_Inv,threshold_Inv,elm);//腐蚀
    cv::imwrite("/storage/emulated/0/opencv_img/erode.png", threshold_Inv);


    Mat element = getStructuringElement(MORPH_RECT ,Size(15 ,3));

    morphologyEx(threshold_Inv ,threshold_Inv,CV_MOP_CLOSE,element);//闭形态学的结构元素

    cv::imwrite("/storage/emulated/0/opencv_img/threshold_Inv.png", threshold_Inv);

    vector< vector <Point> > contours;
    findContours(threshold_Inv ,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//只检测外轮廓

    LOGI("contours size : %d",contours.size());

    //对候选的轮廓进行进一步筛选
    vector< vector <Point> > ::iterator itc = contours.begin();

    while( itc != contours.end())
    {
        RotatedRect mr = minAreaRect(Mat( *itc )); //返回每个轮廓的最小有界矩形区域

        if(!isEligible(mr))  //判断矩形轮廓是否符合要求
        {
            itc = contours.erase(itc);
        }
        else
        {
            rects.push_back(mr);
            ++itc;
        }
    }
}



void normalPosArea(const Mat &intputImg, RotatedRect &rects_optimal, Mat& output_area)
{
    float r,angle;

    angle = rects_optimal.angle;
    r = (float)rects_optimal.size.width / (float) (float)rects_optimal.size.height;
    if(r<1)
        angle = 90 + angle;
    Mat rotmat = getRotationMatrix2D(rects_optimal.center , angle,1);//获得变形矩阵对象
    Mat img_rotated;
    warpAffine(intputImg ,img_rotated,rotmat, intputImg.size(),CV_INTER_CUBIC);

    cv::imwrite("/storage/emulated/0/opencv_img/img_rotated.png", img_rotated);

    //裁剪图像
    Size rect_size = rects_optimal.size;

    if(r<1)
        std::swap(rect_size.width, rect_size.height);
    Mat  img_crop;
    getRectSubPix(img_rotated ,rect_size,rects_optimal.center , img_crop );

    //用光照直方图调整所有裁剪得到的图像，使具有相同宽度和高度，适用于训练和分类
    Mat resultResized;
    resultResized.create(20,300,CV_8UC1);
    cv::resize(img_crop , resultResized,resultResized.size() , 0,0,INTER_CUBIC);

    cv::imwrite("/storage/emulated/0/opencv_img/resultResized.png", resultResized);

    resultResized.copyTo(output_area);

}

void char_segment(const Mat &inputImg, vector<Mat> &dst_mat)
{
    Mat img_threshold;

    Mat whiteImg(inputImg.size(),inputImg.type(),cv::Scalar(255));
    Mat in_Inv = whiteImg - inputImg;

    threshold(in_Inv ,img_threshold , 0,255 ,CV_THRESH_OTSU ); //大津法二值化

    int x_char[19] = {0};
    short counter = 1;
    short num = 0;
    bool *flag = new bool[img_threshold.cols];

    for(int j = 0 ; j < img_threshold.cols;++j)
    {
        flag[j] = true;
        for(int i = 0 ; i < img_threshold.rows ;++i)
        {

            if(img_threshold.at<uchar>(i,j) != 0 )
            {
                flag[j] = false;
                break;
            }

        }
    }

    for(int i = 0;i < img_threshold.cols-2;++i)
    {
        if(flag[i] == true)
        {
            x_char[counter] += i;
            num++;
            if(flag[i+1] ==false && flag[i+2] ==false )
            {
                x_char[counter] = x_char[counter]/num;
                num = 0;
                counter++;
            }
        }
    }
    x_char[18] = img_threshold.cols;

    for(int i = 0;i < 18;i++)
    {
        dst_mat.push_back(Mat(in_Inv , Rect(x_char[i],0, x_char[i+1] - x_char[i] ,img_threshold.rows )));
    }
}

float sumMatValue(const Mat &image)
{
    float sumValue = 0;
    int r = image.rows;
    int c = image.cols;
    if (image.isContinuous())
    {
        c = r*c;
        r = 1;
    }
    for (int i = 0; i < r; i++)
    {
        const uchar* linePtr = image.ptr<uchar>(i);
        for (int j = 0; j < c; j++)
        {
            sumValue += linePtr[j];
        }
    }
    return sumValue;
}

Mat projectHistogram(const Mat &img, int t)
{
    Mat lowData;
    cv::resize(img , lowData ,Size(8 ,16 )); //缩放到8*16

    int sz = (t)? lowData.rows: lowData.cols;
    Mat mhist = Mat::zeros(1, sz ,CV_32F);

    for(int j = 0 ;j < sz; j++ )
    {
        Mat data = (t)?lowData.row(j):lowData.col(j);
        mhist.at<float>(j) = countNonZero(data);
    }

    double min,max;
    minMaxLoc(mhist , &min ,&max);

    if(max > 0)
        mhist.convertTo(mhist ,-1,1.0f/max , 0);

    return mhist;
}

void calcGradientFeat(const Mat &imgSrc, Mat &out)
{
    vector <float>  feat ;
    Mat image;

    //cvtColor(imgSrc,image,CV_BGR2GRAY);
    cv::resize(imgSrc,image,Size(8,16));

    // 计算x方向和y方向上的滤波
    float mask[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

    Mat y_mask = Mat(3, 3, CV_32F, mask) / 8;
    Mat x_mask = y_mask.t(); // 转置
    Mat sobelX, sobelY;

    filter2D(image, sobelX, CV_32F, x_mask);
    filter2D(image, sobelY, CV_32F, y_mask);

    sobelX = abs(sobelX);
    sobelY = abs(sobelY);

    float totleValueX = sumMatValue(sobelX);
    float totleValueY = sumMatValue(sobelY);

    // 将图像划分为4*2共8个格子，计算每个格子里灰度值总和的百分比
    for (int i = 0; i < image.rows; i = i + 4)
    {
        for (int j = 0; j < image.cols; j = j + 4)
        {
            Mat subImageX = sobelX(Rect(j, i, 4, 4));
            feat.push_back(sumMatValue(subImageX) / totleValueX);
            Mat subImageY= sobelY(Rect(j, i, 4, 4));
            feat.push_back(sumMatValue(subImageY) / totleValueY);
        }
    }

    //计算第2个特征
    Mat imageGray;
    //cvtColor(imgSrc,imageGray,CV_BGR2GRAY);
    cv::resize(imgSrc,imageGray,Size(4,8));
    Mat p = imageGray.reshape(1,1);
    p.convertTo(p,CV_32FC1);
    for (int i = 0;i<p.cols;i++ )
    {
        feat.push_back(p.at<float>(i));
    }

    //增加水平直方图和垂直直方图
    Mat vhist = projectHistogram(imgSrc , 1); //水平直方图
    Mat hhist = projectHistogram(imgSrc , 0);  //垂直直方图
    for (int i = 0;i<vhist.cols;i++ )
    {
        feat.push_back(vhist.at<float>(i));
    }
    for (int i = 0;i<hhist.cols;i++ )
    {
        feat.push_back(hhist.at<float>(i));
    }


    out = Mat::zeros(1, feat.size() , CV_32F);
    for (int i = 0;i<feat.size();i++ )
    {
        out.at<float>(i) = feat[i];
    }
}

void getAnnXML() // 此函数仅需运行一次，目的是获得训练矩阵和标签矩阵，保存于ann_xml.xml中
{
    FileStorage fs("ann_xml.xml", FileStorage::WRITE);
    if (!fs.isOpened())
    {
        return;
    }

    Mat  trainData;
    Mat classes = Mat::zeros(1,550,CV_8UC1);   //1700*48维，也即每个样本有48个特征值
    char path[60];
    Mat img_read;
    for (int i = 0;i<10 ;i++)  //第i类
    {
        for (int j=1 ; j< 51 ; ++j)  //i类中第j个，即总共有11类字符，每类有50个样本
        {
            sprintf( path ,"D:\\workspace\\Id_recognition\\Number_char\\%d\\%d (%d).png" , i,i,j);
            img_read = imread(path , 0);
            Mat dst_feature;
            calcGradientFeat(img_read, dst_feature); //计算每个样本的特征矢量
            trainData.push_back(dst_feature);

            classes.at<uchar>(i*50 + j -1) =  i;
        }
    }
    fs<<"TrainingData"<<trainData;
    fs<<"classes"<<classes;
    fs.release();

}


void ann_train(CvANN_MLP &ann, int numCharacters, int nlayers)
{
    Mat trainData ,classes;
    FileStorage fs;
    fs.open("/storage/emulated/0/ann_xml.xml" , FileStorage::READ);

    fs["TrainingData"] >>trainData;
    fs["classes"] >>classes;

    Mat layerSizes(1,3,CV_32SC1);     //3层神经网络
    layerSizes.at<int>( 0 ) = trainData.cols;   //输入层的神经元结点数，设置为24
    layerSizes.at<int>( 1 ) = nlayers; //1个隐藏层的神经元结点数，设置为24
    layerSizes.at<int>( 2 ) = numCharacters; //输出层的神经元结点数为:10
    ann.create(layerSizes , CvANN_MLP::SIGMOID_SYM ,1,1);  //初始化ann
    CvANN_MLP_TrainParams param;
    param.term_crit=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5000,0.01);


    Mat trainClasses;
    trainClasses.create(trainData.rows , numCharacters ,CV_32FC1);
    for (int i =0;i< trainData.rows; i++)
    {
        for (int k=0 ; k< trainClasses.cols ; k++ )
        {
            if ( k == (int)classes.at<uchar> (i))
            {
                trainClasses.at<float>(i,k)  = 1 ;
            }
            else
                trainClasses.at<float>(i,k)  = 0;

        }

    }

    //Mat weights(1 , trainData.rows , CV_32FC1 ,Scalar::all(1) );
    ann.train( trainData ,trainClasses , Mat() ,Mat() ,param);
}

void classify(CvANN_MLP &ann, vector<Mat> &char_Mat, vector<int> &char_result)
{
    char_result.resize(char_Mat.size());
    for (int i=0;i<char_Mat.size(); ++i)
    {
        Mat output(1 ,10 , CV_32FC1); //1*10矩阵

        Mat char_feature;
        calcGradientFeat(char_Mat[i] ,char_feature);
        ann.predict(char_feature ,output);
        Point maxLoc;
        double maxVal;
        minMaxLoc(output , 0 ,&maxVal , 0 ,&maxLoc);

        char_result[i] =  maxLoc.x;

    }
}

void getParityBit(vector<int> &char_result)
{
    int mod = 0;
    int wights[17]={ 7,9,10,5,8,4 ,2,1,6,3,7,9,10,5,8,4,2};
    for(int i =0; i < 17 ;++i)
        mod += char_result[i]*wights[i];//乘相应系数求和

    mod = mod%11; //对11求余

    int value[11]= {1,0,10,9,8,7,6,5,4,3,2};
    char_result[17] = value[mod];

}


JNIEXPORT jstring JNICALL Java_com_opencv_demo_OpenCVHelper_ocr
  (JNIEnv *env, jclass cls,jstring str){

    const char* ch = (char*) (env->GetStringUTFChars(str, 0));
    LOGI("-------------------jstringTostring: %s",ch);
    Mat imgSrc = cv::imread(ch,1);
    Mat imgRplane = getRplane(imgSrc); //获得原始图像R分量
    cv::imwrite("/storage/emulated/0/opencv_img/imgRplane.png", imgRplane);

    LOGI("-------------------getRplane");
    vector <RotatedRect>  rects;
    posDetect(imgRplane ,rects);  //获得身份证号码区域
    LOGI("-------------------posDetect: %d",rects.size());
    Mat outputMat;
    normalPosArea(imgRplane ,rects[0],outputMat); //获得身份证号码字符矩阵
    LOGI("-------------------normalPosArea");
    vector<Mat> char_mat;  //获得切割得的字符矩阵
    char_segment(outputMat , char_mat);
    LOGI("-------------------char_segment");
    //getAnnXML();  //该函数只需执行一次，获得训练矩阵和标签矩阵，保存于xml文件中

    CvANN_MLP ann;
    ann_train(ann, 10, 24); //10为输出层结点,也等于输出的类别，24为隐藏层结点
    LOGI("-------------------ann_train");
    vector<int> char_result;
    classify( ann ,char_mat ,char_result);
    LOGI("-------------------classify");
    getParityBit(char_result); //最后一位易出错，直接由前17位计算最后一位
    LOGI("-------------------getParityBit");
    std::string id = "";
    for(int i = 0; i < char_result.size();++i)
    {
        if (char_result[i] == 10){
            id += "X";
         }else{
            std::stringstream ss;
            ss<<char_result[i];
            id += ss.str();
        }
    }
    const char* id_char = id.c_str();
    jstring js =  env->NewStringUTF(id_char);
    LOGI("-------------------stringTojstring");
    delete id_char;
    return js;
}
