#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <cmath>
#include <cfloat>

/*
    Algorithm plane segmentation
    - Compute surfce normal for all the points
    - cluster normal with same distance d
*/

// checks if the value is nan
template <typename T>
bool is_nan(T d)
{
    if (d > 10.0f || d < 0.01f)
        return true;
    return false;
}

// check if the value from depth image
template <typename T>
bool is_valid_depth(T& depth)
{
    return (depth != std::numeric_limits<T>::min() && depth != std::numeric_limits<T>::max());
}

template <typename T>
T cross_prod(T& a, T&b)
{
    // from cross product of 2 3d vectors
    T res;
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
    return res;
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
    std::map<cv::Point3f,cv::Vec3f > _point_normals;
};

void ImageSeg::filterDepth(cv::Mat& depth)
{
    cv::Mat filtered_depth;
    filtered_depth.setTo(std::numeric_limits<float>::quiet_NaN(), depth == 0);
    depth.convertTo(filtered_depth, CV_32FC1, 1.f/5000.f);

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
    cv::cvtColor(rgb,rgb,CV_BGR2GRAY);
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
    //showImage(_depth_im);
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
    //cv::Mat depth = _depth_im;
    for(int i = 1; i < _cols;i++)
    {
        for(int j = 1;j<_rows;j++)
        {
            float d0 = _depth_im.at<float>(i,j);

            if(!is_nan(d0))
            {
                float d1 = _depth_im.at<float>(i, j-1);
                float d3 = _depth_im.at<float>(i, j+1);
                float d4 = _depth_im.at<float>(i-1,j);
                float d2 = _depth_im.at<float>(i+1,j);
                if(!is_nan(d1) && !is_nan(d2) && !is_nan(d3) && !is_nan(d4))
                {
                    //std::cout << d1<< ", "<<d2<<", "<<d3<<", "<<d4 << std::endl;
                    if (abs(d1-d3) < 0.01 &&  abs(d2-d4) < 0.01)
                    {
                        std::cout << d1<< ", "<<d2<<", "<<d3<<", "<<d4 << std::endl;
                        cv::Vec3f a; // d1-d3
                        a[0] = i -i;
                        a[1] = (j-1) - (j+1);
                        a[2] = d1-d3;
                        cv::Vec3f b; // d2-d3
                        b[0] =  i+1 -(i-1);
                        b[1] = j - j;
                        b[2] = d2-d3;
                        cv::Vec3f normal = cross_prod(b,a); // normal towards camera
                        cv::norm(normal); // normalize
                        cv::Point3f p((float) i,(float) j,d0);
                        //_point_normals.insert(std::pair<cv::Point3f, cv::Vec3f>(p,normal));

                    }
                }
            }
        }
    }

}


int main(int argc, char *argv[]) {
    /* code */
    ImageSeg segment_img;
    segment_img.readDepth(argv[1]);
    segment_img.readRgb("../dataset/image_00004.jpg");
    segment_img.findNormals();
    return 0;
}
