#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <cfloat>

/*
    Algorithm
    - Compute surfce normal for all the points
    -


*/

// checks if the value is nan
template <typename T>
bool is_nan(T d)
{
    return std::numeric_limits<T>::has_quiet_NaN && d != d;
}

// check if the value from depth image
template <typename T>
bool is_valid_depth(T& depth)
{
    return (depth != std::numeric_limits<T>::min() && depth != std::numeric_limits<T>::max());
}

class ImageSeg{
public:
    ImageSeg(){}

    /* @brief: reads image from given filename */
    cv::Mat readImage(std::string filename);

    /* @brief: display image in an external window */
    void showImage(cv::Mat& im);

    /* @brief: intial filtering of depth image */
    void filterDepth(cv::Mat& depth);

    /* @brief: read rgb image  */
    void readRgb(std::string filename);

    /* @brief: read depth image and pre-process it */
    void readDepth(std::string filename);
    /* @brief: computes nearest neigbourhood of the given 3D point  */

    /* @brief: pre-processes depth image and get 3D points */

    /* @brief: calculate normal for the window */
    void findNormals();

private:
    cv::Mat _depth_im; //store depth
    cv::Mat _rgb_im; //store  rgb
    int _rows;  // both should be of same size
    int _cols;  //both should be of same size
};

void ImageSeg::filterDepth(cv::Mat& depth)
{
    cv::Mat filtered_depth;

    depth.convertTo(filtered_depth, CV_32FC1, 1.f/5000.f);
    filtered_depth.setTo(std::numeric_limits<float>::quiet_NaN(), depth == 0);
    filtered_depth.copyTo(depth);
}

/* reads image from file */
cv::Mat ImageSeg::readImage(std::string filename)
{
    cv::Mat im = cv::imread(filename,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
    return im;

}

void ImageSeg::showImage(cv::Mat& im)
{
    cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display", im);
    cv::waitKey(0);
}

void ImageSeg::readRgb(std::string filename)
{
    cv::Mat rgb = readImage(filename);
    rgb.copyTo(_rgb_im);
    _rows = _rgb_im.rows;
    _cols = _rgb_im.cols;

    std::cout<< _cols << ", "<< _rows<<std::endl;
}

void ImageSeg::readDepth(std::string filename)
{
    cv::Mat depth = readImage(filename);
    CV_Assert(depth.type() == CV_16UC1);
    filterDepth(depth);
    depth.copyTo(_depth_im);
    std::cout<<depth.cols << ", "<< _depth_im.rows<<std::endl;


    // filter depth image
    showImage(_depth_im);
    // for(int i = 0; depth_flt.rows;i++)
    //     for(int j = 0; depth_flt.cols;j++)
    //         std::cout << depth_flt.at<float>(i,j)<<std::endl;
    //findNormals(depth_im);

}

void ImageSeg::findNormals()
{
    /*
    1. compute local tangential axis
    2. create cross product for those axis
    */
    cv::Mat depth = _depth_im;
    for(int i = 1; i < _cols;i++)
    {
        for(int j = 1;j<_rows;j++)
        {
            std::cout<< depth.at<float>(i,j)<<std::endl;
        }
    }

}


int main(int argc, char *argv[]) {
    /* code */
    ImageSeg segment_img;
    segment_img.readDepth(argv[1]);
    segment_img.readRgb(argv[2]);
    //segment_img.findNormals();
    return 0;
}
