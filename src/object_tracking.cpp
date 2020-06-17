#include <stdlib.h>
#include <functional>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.h"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/image_encodings.hpp"

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>

class ImageConverter : public rclcpp::Node {
private:
    const std::string OPENCV_WINDOW = "Image window";
//    image_transport::ImageTransport it;
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_;
    cv::Mat roi, hsv_roi, mask, roi_hist;
    cv::Mat * lastFrame;

    float range_[2] = {0, 180};
    int histSize[1] = {180};
    int channels[1] = {0};
    const float *range[1] = {range_};

    cv::Rect track_window;
    cv::TermCriteria term_crit;

    void trackObjectAndDisplay( cv::Mat& frame )
    {
        if (lastFrame == NULL) {

            // setup initial location of window
            track_window = cv::Rect(frame.cols/2-50, frame.rows/2-50, 100, 100); // simply hardcoded the values

            try {
                // set up the ROI for tracking
                roi = frame(track_window);
                cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
                cv::inRange(hsv_roi, cv::Scalar(0, 60, 32), cv::Scalar(180, 255, 255), mask);

                cv::calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
                cv::normalize(roi_hist, roi_hist, 0, 255, cv::NORM_MINMAX);
            } catch (cv::Exception err) {
                std::cout << err.msg << std::endl;
            }

            // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
            term_crit = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);
        }

        cv::Mat hsv, dst;
        cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        // apply camshift to get the new location
        cv::RotatedRect rot_rect = CamShift(dst, track_window, term_crit);

        // Draw it on image
        cv::Point2f points[4];
        rot_rect.points(points);
        for (int i = 0; i < 4; i++)
            line(frame, points[i], points[(i+1)%4], 255, 2);

        imshow(OPENCV_WINDOW, frame );

        lastFrame = &frame;
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

        trackObjectAndDisplay(cv_ptr->image);
        cv::waitKey(3);

        pub_.publish(cv_ptr->toImageMsg());
    }

public:
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    ImageConverter() : Node("image_converter") {
//    ImageConverter() : Node("image_converter"), it(this->get()) {

        // Open demo window that will show output image
        cv::namedWindow(OPENCV_WINDOW);

        cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
        cv::String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

        //-- 1. Load the cascades
        if( !face_cascade.load( face_cascade_name ) )
        {
            std::cout << "--(!)Error loading face cascade\n";
        };
        if( !eyes_cascade.load( eyes_cascade_name ) )
        {
            std::cout << "--(!)Error loading eyes cascade\n";
        };

        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
        pub_ = image_transport::create_publisher(this, "out_image_base_topic", custom_qos);
        sub_ = image_transport::create_subscription(this, "in_image_base_topic",
                std::bind(&ImageConverter::imageCallback, this, std::placeholders::_1), "raw", custom_qos);

    }

    ~ImageConverter()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConverter>());
    rclcpp::shutdown();
    return 0;
}