#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>

#include <modelify/alignment_toolbox/alignment_toolbox.h>
#include <modelify/common.h>
#include <modelify/matching_visualizer.h>
#include <modelify/pcl_tools.h>
#include <modelify/rgbd_tools.h>

using namespace modelify;

struct Dataset {
  std::vector<cv::Mat> image;
  std::vector<cv::Mat> image_rect;
  std::vector<cv::Mat> image_rect_reg;
};

template <typename M>
class BagSubscriber : public message_filters::SimpleFilter<M> {
 public:
  void newMessage(const boost::shared_ptr<M const>& msg) {
    this->signalMessage(msg);
  }
};

void visualize(const PointSurfelCloudType::ConstPtr cloud1,
               const PointSurfelCloudType::ConstPtr cloud2,
               const std::string& test_name) {
  std::unique_ptr<MatchingVisualizer> visualizer;
  constexpr bool kShowXWindow = true;
  visualizer.reset(new MatchingVisualizer(test_name, kShowXWindow));
  // visualizer->setFileName("test_results/" + test_name + ".png");
  visualizer->setPointClouds(cloud1, cloud2);
  visualizer->visualize();
}

void callback(const sensor_msgs::Image::ConstPtr& image_message,
              const sensor_msgs::Image::ConstPtr& image_rect_message,
              const sensor_msgs::Image::ConstPtr& image_rect_reg_message,
              const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info,
              const sensor_msgs::PointCloud2::ConstPtr& pointcloud_message,
              Dataset* data) {
  std::cout << "Hello, marko here!" << std::endl;

  cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(
      image_message, sensor_msgs::image_encodings::TYPE_32FC1);
  cv_bridge::CvImagePtr cv_image_rect = cv_bridge::toCvCopy(
      image_rect_message, sensor_msgs::image_encodings::TYPE_32FC1);
  cv_bridge::CvImagePtr cv_image_rect_reg = cv_bridge::toCvCopy(
      image_rect_reg_message, sensor_msgs::image_encodings::TYPE_32FC1);

  cv::Mat image = cv_image->image;
  image.convertTo(image, CV_16UC1, 32767);
  cv::imwrite("/home/panjekm/asl_work/camera_calibration_1306030063/image.png",
              image);

  cv::Mat image_rect = cv_image_rect->image;
  image_rect.convertTo(image_rect, CV_16UC1, 32767);
  cv::imwrite(
      "/home/panjekm/asl_work/camera_calibration_1306030063/image_rect.png",
      image_rect);

  cv::Mat image_rect_reg = cv_image_rect_reg->image;
  image_rect_reg.convertTo(image_rect_reg, CV_16UC1, 32767);
  cv::imwrite(
      "/home/panjekm/asl_work/camera_calibration_1306030063/image_rect_reg.png",
      image_rect_reg);

  cv::Mat diff_image = cv_image_rect->image - cv_image_rect_reg->image;
  diff_image.convertTo(diff_image, CV_16UC1, 32767);
  cv::imwrite("/home/panjekm/asl_work/camera_calibration_1306030063/diff.png",
              diff_image);

  PointSurfelCloudType::Ptr depth_cloud(new PointSurfelCloudType);
  cv::Mat depth_intrinsics =
      cv::Mat(3, 3, CV_64FC1, const_cast<double*>(&depth_camera_info->K[0]));
  projectDepthTo3D(cv_image_rect->image, depth_intrinsics, depth_cloud);

  PointSurfelCloudType::Ptr depth_cloud_driver(new PointSurfelCloudType);
  pcl::PCLPointCloud2 pcl_pointcloud2;
  pcl_conversions::toPCL(*pointcloud_message, pcl_pointcloud2);
  pcl::fromPCLPointCloud2(pcl_pointcloud2, *depth_cloud_driver);

  visualize(depth_cloud, depth_cloud_driver, "depth cloud");

  LOG(ERROR) << "Done.";
};

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  ros::Time::init();

  Dataset data;

  rosbag::Bag bag;
  bag.open("/home/panjekm/asl_work/camera_calibration_1306030063/capture.bag",
           rosbag::bagmode::Read);

  std::vector<std::string> topics;

  const std::string kTopicImage = "/camera/depth/image";
  const std::string kTopicImageRect = "/camera/depth/image_rect";
  const std::string kTopicImageRectReg =
      "/camera/depth_registered/sw_registered/image_rect";
  const std::string kTopicDepthCameraInfo = "/camera/depth/camera_info";
  const std::string kTopicPointcloud = "/camera/depth/points";

  topics.push_back(kTopicImage);
  topics.push_back(kTopicImageRect);
  topics.push_back(kTopicImageRectReg);
  topics.push_back(kTopicDepthCameraInfo);
  topics.push_back(kTopicPointcloud);

  rosbag::View view(bag, rosbag::TopicQuery(topics));

  BagSubscriber<sensor_msgs::Image> sub_image;
  BagSubscriber<sensor_msgs::Image> sub_image_rect;
  BagSubscriber<sensor_msgs::Image> sub_image_rect_reg;
  BagSubscriber<sensor_msgs::CameraInfo> sub_depth_camera_info;
  BagSubscriber<sensor_msgs::PointCloud2> sub_pointcloud;

  constexpr size_t kMessageQueueSize = 25u;

  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image,
                                    sensor_msgs::Image, sensor_msgs::CameraInfo,
                                    sensor_msgs::PointCloud2>
      sync(sub_image, sub_image_rect, sub_image_rect_reg, sub_depth_camera_info,
           sub_pointcloud, kMessageQueueSize);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, &data));
  for (const rosbag::MessageInstance& message : view) {
    sensor_msgs::Image::ConstPtr image_msg =
        message.instantiate<sensor_msgs::Image>();
    std::string message_topic = message.getTopic();
    if (image_msg.get() != nullptr) {
      if (message_topic.compare(kTopicImage) == 0) {
        sub_image.newMessage(image_msg);
      } else if (message_topic.compare(kTopicImageRect) == 0) {
        sub_image_rect.newMessage(image_msg);
      } else if (message_topic.compare(kTopicImageRectReg) == 0) {
        sub_image_rect_reg.newMessage(image_msg);
      }
    }
    sensor_msgs::CameraInfo::ConstPtr camera_info_msg =
        message.instantiate<sensor_msgs::CameraInfo>();
    if (camera_info_msg.get() != nullptr) {
      if (message_topic.compare(kTopicDepthCameraInfo) == 0) {
        sub_depth_camera_info.newMessage(camera_info_msg);
      }
    }
    sensor_msgs::PointCloud2::ConstPtr pointcloud_msg =
        message.instantiate<sensor_msgs::PointCloud2>();
    if (pointcloud_msg.get() != nullptr) {
      if (message_topic.compare(kTopicPointcloud) == 0) {
        sub_pointcloud.newMessage(pointcloud_msg);
      }
    }
  }

  bag.close();

  return 1;
}
