#include <iostream>
#include <fstream>
#include <boost/config.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <map>
#include <cmath>
#include <cfloat>
#include <random>

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
template <typename T>
void normalize(T& v)
{
    float s = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] = v[0]/s;
    v[1] = v[1]/s;
    v[2] = v[2]/s;

}

template <typename T>
bool match(T& a, T& b)
{
    float dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    //std::cout << dot<<std::endl;
    if (dot >= 0.99999f)
        return true;
    return false;
    //return (a[0] != b[0] || a[1] != b[1] || a[2] != b[2] );
}

struct Node{
    int i;
    int j;
};

// undirected edge between a and b with weight w
struct Weight{
    float w;
};



class ImageSeg{
    typedef boost::adjacency_list<boost::vecS,boost::vecS, boost::directedS, Node, Weight> Graph;
    //typedef boost::adjacency_list_traits < boost::vecS, boost::vecS, boost::directedS, Node, Weight > Traits;
    //typedef boost::property_map<Graph, boost::edge_weight_t>::type weight_map_type;
    //typedef boost::property_traits<weight_map_type>::value_type weight_type;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
	typedef boost::graph_traits<Graph>::edge_descriptor Edge;
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

    void minCutSegmentation();

    void addVertex(Graph& g, int i, int j );

    void addEdge(Graph& g,Vertex& v1, Vertex& v2, float w);

    void watershedSeg();


private:
    cv::Mat _depth_im; //store depth
    cv::Mat _rgb_im; //store  rgb
    int _rows;  // both should be of same size
    int _cols;  //both should be of same size
    std::map<cv::Point3f,cv::Vec3f > _point_normals;
    Graph _g;
};

void ImageSeg::filterDepth(cv::Mat& depth)
{
    cv::Mat filtered_depth;
    filtered_depth.setTo(std::numeric_limits<float>::quiet_NaN(), depth == 0);

    depth.convertTo(filtered_depth, CV_32FC1, 1.f/5000.f);
    // filter depth image for better estimate
    cv::bilateralFilter(filtered_depth, depth, -1, 0.03, 4.5);
    //filtered_depth.copyTo(depth);
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
    //cv::cvtColor(rgb,rgb,CV_BGR2GRAY);
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
    cv::Mat_<cv::Vec3f> normal_image(_cols,_rows, CV_16UC1);
    cv::Mat plane = cv::Mat::zeros(_rows,_cols, CV_16UC1);
    std::vector<cv::Vec3f> normal_clustor;
    std::vector<std::vector<std::pair<int, int> > > pixel_bins;
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
                    if (std::abs(d1-d3) < 0.01 &&  std::abs(d2-d4) < 0.01)
                    {
                        //std::cout << d1<< ", "<<d2<<", "<<d3<<", "<<d4 << std::endl;
                        cv::Vec3f a; // d1-d3
                        a[0] = i -i;
                        a[1] = (j-1) - (j+1);
                        a[2] = d1-d3;
                        cv::Vec3f b; // d2-d3
                        b[0] =  i+1 -(i-1);
                        b[1] = j - j;
                        b[2] = d2-d3;
                        cv::Vec3f normal = cross_prod(b,a); // normal towards camera
                        normalize(normal);
                        //cv::norm(normal); // normalize
                        //cv::Point3f p((float) i,(float) j,d0);
                        plane.at<int>(i,j) = (int) (255.0f * normal[2]);

                        if(normal_clustor.empty())
                        {
                            normal_clustor.push_back(normal);
                            std::pair<int, int> curr_pixel(i,j);
                            std::vector<std::pair<int, int> > curr_bin;
                            curr_bin.push_back(curr_pixel);
                            pixel_bins.push_back(curr_bin);



                        }else{
                            bool found = false;

                            for(int k = 0; k < normal_clustor.size();k++)
                            {
                                if(match(normal, normal_clustor[k]))
                                {
                                    //std::cout << "k :"<< k << std::endl;
                                    found = true;
                                    std::pair<int,int> curr_pixel(i,j);
                                    pixel_bins[k].push_back(curr_pixel);
                                    break;
                                }
                            }
                            if(!found)
                            {
                                normal_clustor.push_back(normal);
                                std::pair<int,int> curr_pixel(i,j);
                                std::vector<std::pair<int, int> > curr_bin;
                                curr_bin.push_back(curr_pixel);
                                pixel_bins.push_back(curr_bin);
                            }
                        }

                        normal_image.at<cv::Vec3f>(i,j) = normal;
                        //std::cout << normal[0]<< ", "<<normal[1]<<", "<<normal[2]<< std::endl;
                        //_rgb_im.at<int>(i,j)= 0;


                    }
                }
            }
        }
    }


    /*
    for (int i =0 ; i < pixel_bins[1].size(); i++)
    {
        plane.at<int>(pixel_bins[0][i].first, pixel_bins[0][i].second) = 255;
        _rgb_im.at<int>(pixel_bins[0][i].first, pixel_bins[0][i].second) = 255;
    }
    for(int i = 0; i < pixel_bins.size();i++)
    {
        std::cout << pixel_bins[i].size()<< std::endl;
    }
    std::cout<< pixel_bins.size()<<std::endl;
    */
    showImage(plane);;

}


void ImageSeg::addVertex(Graph& g, int i, int j)
{
    Vertex v ;
    v = boost::add_vertex(g);
    g[v].i = i;
    g[v].j = j;
}


void ImageSeg::addEdge(Graph& g, Vertex& v1, Vertex& v2, float w)
{
    Edge e = (boost::add_edge(v1,v2,g)).first;
    g[e].w = w;
}


void ImageSeg::minCutSegmentation()
{
    /* Algorithm
        1. create graph

    */

    cv::Mat_<int> dissimilar = cv::Mat::zeros(_rows,_cols, CV_16UC1);

    for(int i =0; i < _rows;i++)
    {
        for (int j = 0; j <_cols;j++)
        {
            addVertex(_g, i,j);
        }
    }

    // create edges
    Graph::vertex_iterator vertex_It,vertex_End, tmp_it;
    boost::tie(vertex_It, vertex_End) = boost::vertices(_g);
    for(;vertex_It!=vertex_End;++vertex_It)
    {
        Vertex v1 = *vertex_It;
        int x =  _g[v1].i;
        int y = _g[v1].j;

        for(int i = 1;  i < 5; i++ )
        {
            Vertex v2 = *(vertex_It+i);
            if(*(vertex_It+i))
            {

                int x1 = _g[v2].i;
                int y1 = _g[v2].j;
                int val1 = _rgb_im.at<int>(x,y);
                int val2 = _rgb_im.at<int>(x1,y1);
                float d1 = _depth_im.at<float>(x,y);
                float d2 = _depth_im.at<float>(x1,y1);
                if(is_nan(d1) && is_nan(d2))
                {
                    float dist = (float) std::abs(val1-val2);
                    dissimilar.at<int>(x1,y1) =(int) 255.0f*dist;
                    addEdge(_g,v1, v2, dist);
                    continue;
                }
                float dist = std::abs(val1 - val2) + std::abs(d1-d2);
                dissimilar.at<int>(x1,y1) =(int) 255.0f*dist;
                addEdge(_g,v1, v2, dist);
            }
        }


    }

    std::vector<boost::default_color_type> color(boost::num_vertices(_g));
    std::vector<long> distance(boost::num_vertices(_g));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, boost::num_vertices(_g)-1);
    boost::tie(vertex_It, vertex_End) = boost::vertices(_g);

    //long flow = boost::boykov_kolmogorov_max_flow(_g ,*vertex_It, *(vertex_It+5));

    showImage(dissimilar);

}


void ImageSeg::watershedSeg()
{
    // filter using laplacian kernel
    cv::Mat kernel = (cv::Mat_<float>(3,3) <<
        0.1, 0.1, 0.1,
        0.1, -0.8, 0.1,
        0.1, 0.1, 0.1);

    cv::Mat src = _rgb_im;
    cv::cvtColor(_rgb_im,_rgb_im,CV_BGR2GRAY);
    cv::Mat sharp = _rgb_im;
    cv::Mat laplacianImg;
    cv::filter2D(sharp, laplacianImg, CV_8UC1, kernel);

    _rgb_im.convertTo(sharp, CV_8U);
    cv::Mat resultImg = sharp - laplacianImg;
    
    cv::threshold(resultImg, resultImg, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    
    cv::Mat dist;
    cv::distanceTransform(resultImg, dist, CV_DIST_L2,3);
    cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
    cv::threshold(dist, dist, .2, 1., CV_THRESH_BINARY);

    cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8UC1);
    cv::dilate(dist, dist, kernel1);

    cv::Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32SC1);

    for (size_t i = 0; i < contours.size(); i++)
        cv::drawContours(markers, contours, static_cast<int>(i), cv::Scalar::all(static_cast<int>(i)+1), -1);

    cv::circle(markers, cv::Point(5,5), 3, CV_RGB(255,255,255), -1);
    markers = markers*10000;

    cv::watershed(src, markers);
    cv::Mat mark = cv::Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    cv::bitwise_not(mark, mark);

    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = cv::theRNG().uniform(0, 255);
        int g = cv::theRNG().uniform(0, 255);
        int r = cv::theRNG().uniform(0, 255);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);

    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<cv::Vec3b>(i,j) = colors[index-1];
            else
                dst.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
        }
    }

    showImage(markers);
}


int main(int argc, char *argv[]) {
    /* code */
    ImageSeg segment_img;
    segment_img.readDepth("../dataset/depth_00004.pgm");
    segment_img.readRgb("../dataset/image_00001.jpg");
    //segment_img.findNormals();
    //segment_img.minCutSegmentation();
    segment_img.watershedSeg();
    return 0;
}
