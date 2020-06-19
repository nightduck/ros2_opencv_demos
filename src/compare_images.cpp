#include <stdlib.h>
#include <functional>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.h"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/image_encodings.hpp"

#include <opencv2/core.hpp>      // Basic OpenCV structures
#include <opencv2/core/utility.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

// CUDA structures and methods
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

struct BufferMSSIM                                     // Optimized CUDA versions
{   // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    cv::cuda::GpuMat gI1, gI2, gs, t1,t2;

    cv::cuda::GpuMat I1_2, I2_2, I1_I2;
    std::vector<cv::cuda::GpuMat> vI1, vI2;

    cv::cuda::GpuMat mu1, mu2;
    cv::cuda::GpuMat mu1_2, mu2_2, mu1_mu2;

    cv::cuda::GpuMat sigma1_2, sigma2_2, sigma12;
    cv::cuda::GpuMat t3;

    cv::cuda::GpuMat ssim_map;

    cv::cuda::GpuMat buf;
};

class ImageConverter : public rclcpp::Node {
private:
//    image_transport::ImageTransport it;
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_;
    cv::Mat prev;

    cv::Scalar getMSSIM_CUDA_optimized( const cv::Mat& i1, const cv::Mat& i2, BufferMSSIM& b)
    {
        const float C1 = 6.5025f, C2 = 58.5225f;
        /***************************** INITS **********************************/

        b.gI1.upload(i1);
        b.gI2.upload(i2);

        cv::cuda::Stream stream;

        b.gI1.convertTo(b.t1, CV_32F, stream);
        b.gI2.convertTo(b.t2, CV_32F, stream);

        cv::cuda::split(b.t1, b.vI1, stream);
        cv::cuda::split(b.t2, b.vI2, stream);
        cv::Scalar mssim;

        cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(b.vI1[0].type(), -1, cv::Size(11, 11), 1.5);

        for( int i = 0; i < b.gI1.channels(); ++i )
        {
            cv::cuda::multiply(b.vI2[i], b.vI2[i], b.I2_2, 1, -1, stream);        // I2^2
            cv::cuda::multiply(b.vI1[i], b.vI1[i], b.I1_2, 1, -1, stream);        // I1^2
            cv::cuda::multiply(b.vI1[i], b.vI2[i], b.I1_I2, 1, -1, stream);       // I1 * I2

            gauss->apply(b.vI1[i], b.mu1, stream);
            gauss->apply(b.vI2[i], b.mu2, stream);

            cv::cuda::multiply(b.mu1, b.mu1, b.mu1_2, 1, -1, stream);
            cv::cuda::multiply(b.mu2, b.mu2, b.mu2_2, 1, -1, stream);
            cv::cuda::multiply(b.mu1, b.mu2, b.mu1_mu2, 1, -1, stream);

            gauss->apply(b.I1_2, b.sigma1_2, stream);
            cv::cuda::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cv::cuda::GpuMat(), -1, stream);
            //b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation

            gauss->apply(b.I2_2, b.sigma2_2, stream);
            cv::cuda::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cv::cuda::GpuMat(), -1, stream);
            //b.sigma2_2 -= b.mu2_2;

            gauss->apply(b.I1_I2, b.sigma12, stream);
            cv::cuda::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cv::cuda::GpuMat(), -1, stream);
            //b.sigma12 -= b.mu1_mu2;

            //here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
            cv::cuda::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); //b.t1 = 2 * b.mu1_mu2 + C1;
            cv::cuda::add(b.t1, C1, b.t1, cv::cuda::GpuMat(), -1, stream);
            cv::cuda::multiply(b.sigma12, 2, b.t2, 1, -1, stream); //b.t2 = 2 * b.sigma12 + C2;
            cv::cuda::add(b.t2, C2, b.t2, cv::cuda::GpuMat(), -12, stream);

            cv::cuda::multiply(b.t1, b.t2, b.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

            cv::cuda::add(b.mu1_2, b.mu2_2, b.t1, cv::cuda::GpuMat(), -1, stream);
            cv::cuda::add(b.t1, C1, b.t1, cv::cuda::GpuMat(), -1, stream);

            cv::cuda::add(b.sigma1_2, b.sigma2_2, b.t2, cv::cuda::GpuMat(), -1, stream);
            cv::cuda::add(b.t2, C2, b.t2, cv::cuda::GpuMat(), -1, stream);

            cv::cuda::multiply(b.t1, b.t2, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
            cv::cuda::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;

            stream.waitForCompletion();

            cv::Scalar s = cv::cuda::sum(b.ssim_map, b.buf);
            mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);
        }
        return mssim;
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        BufferMSSIM b;
        cv::Scalar s = getMSSIM_CUDA_optimized(cv_ptr->image, prev, b);

        std::cout << s.val[0] << ", " << s.val[1] << ", " << s.val[2] << ", " << s.val[3] << std::endl;

        prev = cv_ptr->image.clone();
    }

public:
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    ImageConverter() : Node("compare_images") {
//    ImageConverter() : Node("compare_images"), it(this->get()) {

        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
        sub_ = image_transport::create_subscription(this, "in_image_base_topic",
                std::bind(&ImageConverter::imageCallback, this, std::placeholders::_1), "raw", custom_qos);

//        pub = it.advertise("out_image_base_topic", 1);
//        sub = it.subscribe("in_image_base_topic", 1, imageCallback);
    }

    ~ImageConverter() {}
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConverter>());
    rclcpp::shutdown();
    return 0;
}