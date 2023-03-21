#include "frontend/ImageRW.h"

#include <opencv4/opencv2/core/version.hpp>

#if (defined(CV_VERSION_EPOCH) && CV_VERSION_EPOCH == 2)
#include <opencv4/opencv2/highgui.hpp>
#else
#include <opencv4/opencv2/imgcodecs.hpp>
#endif

namespace ldso
{
    namespace IOWrap
    {
        MinimalImageB *readImageBW_8U(std::string filename)
        {
            cv::Mat m = cv::imread(filename,cv::IMREAD_GRAYSCALE);

            if (m.rows * m.cols == 0)
            {
                LOG(ERROR) << "cv::imread could not read image" << filename << "! this may segfault." << endl;
                return 0;
            }

            if (m.type() != CV_8U)
            {
                LOG(ERROR) << "cv::imread did something strange! this may segfault." << endl;
                return 0;
            }

            MinimalImageB *img = new MinimalImageB(m.cols, m.rows);
            memcpy(img->data, m.data, m.rows * m.cols);
            return img;
        }

        MinimalImageB3 *readImageRGB_8U(std::string filename)
        {
            cv::Mat m = cv::imread(filename,cv::IMREAD_COLOR);

            if (m.rows * m.cols == 0)
            {
                LOG(ERROR) << "cv::imread could not read image" << filename << "! this may segfault." << endl;
                return 0;
            }

            if (m.type() != CV_8UC3)
            {
                LOG(ERROR) << "cv::imread did something strange! this may segfault." << endl;
                return 0;
            }

            MinimalImageB3 *img = new MinimalImageB3(m.cols, m.rows);
            memcpy(img->data, m.data, 3 * m.rows * m.cols);
            return img;
        }

        MinimalImage<unsigned short> *readImageBW_16U(std::string filename)
        {
            cv::Mat m = cv::imread(filename,cv::IMREAD_UNCHANGED);

            if (m.rows * m.cols == 0)
            {
                LOG(ERROR) << "cv::imread could not read image" << filename << "! this may segfault." << endl;
                return 0;
            }

            if (m.type() != CV_16U)
            {
                LOG(ERROR) << "readImageBW_16U called on image that is not a 16bit grayscale image. this may segfault." << endl;
                return 0;
            }

            MinimalImage<unsigned short> *img = new MinimalImage<unsigned short>(m.cols, m.rows);
            memcpy(img->data, m.data, 2 * m.rows * m.cols);
            return img;
        }

        MinimalImageB *readStreamBW_8U(char *data, int numBytes)
        {
            cv::Mat m = cv::imdecode(cv::Mat(numBytes, 1, CV_8U, data), cv::IMREAD_GRAYSCALE);

            if (m.rows * m.cols == 0)
            {
                LOG(ERROR) << "cv::imdecode could not read stream (" << numBytes << " bytes)! this may segfault." << endl;
                return 0;
            }

            if (m.type() != CV_8U)
            {
                LOG(ERROR) << "cv::imdecode did something strange! this may segfault." << endl;
                return 0;
            }

            MinimalImageB *img = new MinimalImageB(m.cols, m.rows);
            memcpy(img->data, m.data, m.rows * m.cols);
            return img;
        }


        void writeImage(std::string filename, MinimalImageB *img){
            cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8U, img->data));
        }

        void writeImage(std::string filename, MinimalImageB3 *img){
            cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8UC3, img->data));

        }

        void writeImage(std::string filename, MinimalImageF *img){
            cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32F, img->data));

        }

        void writeImage(std::string filename, MinimalImageF3 *img){
            cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32FC3, img->data));
        }


    } // namespace IOWrap

} // namespace ldso




