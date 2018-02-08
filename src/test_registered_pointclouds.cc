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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

#include <modelify/alignment_toolbox/alignment_toolbox.h>
#include <modelify/common.h>
#include <modelify/matching_visualizer.h>
#include <modelify/pcl_tools.h>
#include <modelify/rgbd_tools.h>

#include "modelify_tools/depth_traits.h"

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

void convertPointCloudToMat(const PointSurfelCloudType::ConstPtr pointcloud,
                            cv::Mat& cv_mat) {
  for (PointSurfelType point : *pointcloud) {
    cv::Vec3f cv_point;
    cv_point[0] = point.x;
    cv_point[1] = point.y;
    cv_point[2] = point.z;
    cv_mat.push_back(cv_point);
  }
}

void convertMatToPointCloud(const cv::Mat& cv_mat,
                            PointSurfelCloudType::Ptr pointcloud) {
  std::cout << "MARKO " << cv_mat.rows << std::endl;
  for (size_t i = 0; i < cv_mat.rows; ++i) {
    PointSurfelType point;
    cv::Vec3f cv_point = cv_mat.at<cv::Vec3f>(i);
    point.x = cv_point[0];
    point.y = cv_point[1];
    point.z = cv_point[2];
    pointcloud->push_back(point);
  }
}

template <typename T>
void registerImage(const sensor_msgs::ImageConstPtr& depth_msg,
                   const sensor_msgs::ImagePtr& registered_msg,
                   const Eigen::Affine3d& depth_to_rgb,
                   PointSurfelCloudType::Ptr pcl_pointcloud) {
  // Allocate memory for registered depth image
  registered_msg->step = registered_msg->width * sizeof(T);
  registered_msg->data.resize(registered_msg->height * registered_msg->step);
  // data is already zero-filled in the uint16 case, but for floats we want to
  // initialize everything to NaN.
  DepthTraits<T>::initializeBuffer(registered_msg->data);

  // Extract all the parameters we need
  double inv_depth_fx = 1.0 / 574.8481997286975;
  double inv_depth_fy = 1.0 / 574.7309430753069;
  double depth_cx = 314.3315246963823;
  double depth_cy = 239.174665616155;

  double rgb_fx = 537.8710412230362;
  double rgb_fy = 537.8046290358968;
  double rgb_cx = 319.27400522706114;
  double rgb_cy = 238.42134955498372;

  // Transform the depth values into the RGB frame
  /// @todo When RGB is higher res, interpolate by rasterizing depth triangles
  /// onto the registered image
  const T* depth_row = reinterpret_cast<const T*>(&depth_msg->data[0]);
  int row_step = depth_msg->step / sizeof(T);
  T* registered_data = reinterpret_cast<T*>(&registered_msg->data[0]);
  int raw_index = 0;
  for (unsigned v = 0; v < depth_msg->height; ++v, depth_row += row_step) {
    for (unsigned u = 0; u < depth_msg->width; ++u, ++raw_index) {
      T raw_depth = depth_row[u];
      if (!DepthTraits<T>::valid(raw_depth)) continue;

      /// @todo Combine all operations into one matrix multiply on (u,v,d)
      // Reproject (u,v,Z) to (X,Y,Z,1) in depth camera frame
      Eigen::Vector4d xyz_depth;
      xyz_depth << ((u - depth_cx) * raw_depth) * inv_depth_fx,
          ((v - depth_cy) * raw_depth) * inv_depth_fy, raw_depth, 1;

      // Transform to RGB camera frame
      Eigen::Vector4d xyz_rgb = depth_to_rgb * xyz_depth;
      PointSurfelType point;
      point.x = xyz_rgb[0];
      point.y = xyz_rgb[1];
      point.z = xyz_rgb[2];
      pcl_pointcloud->push_back(point);

      // Project to (u,v) in RGB image
      double inv_Z = 1.0 / xyz_rgb.z();
      int u_rgb = (rgb_fx * xyz_rgb.x()) * inv_Z + rgb_cx + 0.5;
      int v_rgb = (rgb_fy * xyz_rgb.y()) * inv_Z + rgb_cy + 0.5;

      if (u_rgb < 0 || u_rgb >= (int)registered_msg->width || v_rgb < 0 ||
          v_rgb >= (int)registered_msg->height)
        continue;

      T& reg_depth = registered_data[v_rgb * registered_msg->width + u_rgb];
      T new_depth = xyz_rgb.z();
      // Validity and Z-buffer checks
      if (!DepthTraits<T>::valid(reg_depth) || reg_depth > new_depth)
        reg_depth = new_depth;
    }
  }
}

void projectPointsToImage(const cv::Mat& pointcloud, const cv::Mat& intrinsics,
                          cv::Mat* image) {
  CHECK(!intrinsics.empty());
  CHECK_EQ(3, intrinsics.rows);
  CHECK_EQ(3, intrinsics.cols);
  CHECK_NOTNULL(image);
  CHECK_EQ(image->type(), CV_32FC1);

  float fx = intrinsics.at<float>(0, 0);
  float fy = intrinsics.at<float>(1, 1);
  float cx = intrinsics.at<float>(0, 2);
  float cy = intrinsics.at<float>(1, 2);

  for (size_t i = 0u; i < pointcloud.rows; ++i) {
    cv::Vec3f point = pointcloud.at<cv::Vec3f>(i);

    float inv_z = 1.0 / point[2];
    size_t u = (fx * point[0]) * inv_z + cx + 0.5;
    size_t v = (fy * point[1]) * inv_z + cy + 0.5;

    // Check if we are within image bounds.
    if (u < 0 || u >= image->cols || v < 0 || v >= image->rows) {
      continue;
    }
    float current_depth = image->at<float>(v, u);
    float new_depth = point[2];

    // Only update if depth doesnt exist or is smaller than before.
    if (current_depth == 0.0 || current_depth > new_depth) {
      image->at<float>(v, u) = new_depth;
    }
  }
}

void callback(
    const sensor_msgs::Image::ConstPtr& image_message,
    const sensor_msgs::Image::ConstPtr& image_rect_message,
    const sensor_msgs::Image::ConstPtr& image_rect_reg_message,
    const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info,
    const sensor_msgs::PointCloud2::ConstPtr& pointcloud_message,
    const sensor_msgs::PointCloud2::ConstPtr& pointcloud_registered_message,
    Dataset* data) {
  std::cout << "Hello, marko here!" << std::endl;

  cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(
      image_message, sensor_msgs::image_encodings::TYPE_32FC1);
  cv_bridge::CvImagePtr cv_image_rect = cv_bridge::toCvCopy(
      image_rect_message, sensor_msgs::image_encodings::TYPE_32FC1);
  cv_bridge::CvImagePtr cv_image_rect_reg = cv_bridge::toCvCopy(
      image_rect_reg_message, sensor_msgs::image_encodings::TYPE_32FC1);

  cv::Mat depth_intrinsics =
      cv::Mat(3, 3, CV_64FC1, const_cast<double*>(&depth_camera_info->K[0]));
  cv::Mat depth_distortion =
      cv::Mat(1, 5, CV_64FC1, const_cast<double*>(&depth_camera_info->D[0]));
  cv::Mat depth_intrinsics_P =
      cv::Mat(3, 4, CV_64FC1, const_cast<double*>(&depth_camera_info->P[0]));
  cv::Mat depth_intrinsics_R =
      cv::Mat(3, 3, CV_64FC1, const_cast<double*>(&depth_camera_info->R[0]));

  double rgb_intrinsics_data_calibrated[9] = {537.8710412230362,
                                              0.0,
                                              319.27400522706114,
                                              0.0,
                                              537.8046290358968,
                                              238.42134955498372,
                                              0.0,
                                              0.0,
                                              1.0};
  double rgb_distortion_data_calibrated[5] = {
      0.03615361344570411, -0.11649135824747184, 0.0002728867477055461,
      0.00017860148085525662, 0.0};

  double rgb_intrinsics_data[9] = {574.0527954101562,
                                   0.0,
                                   319.5,
                                   0.0,
                                   574.0527954101562,
                                   239.5,
                                   0.0,
                                   0.0,
                                   1.0};
  double rgb_distortion_data[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  double extrinsics_data_calibrated[16] = {0.999967840215677,
                                           0.007498511480668,
                                           0.002844443701247,
                                           -0.035642737250000,
                                           -0.007494243770878,
                                           0.999970780590298,
                                           -0.001508070267716,
                                           0.000233876573000,
                                           -0.002855668870497,
                                           0.001486704814012,
                                           0.999994817418620,
                                           -0.000359037985000,
                                           0,
                                           0,
                                           0,
                                           1};
  double extrinsics_data[16] = {1.0, 0.0, 0.0, 0.025, 0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,   0.0, 0.0, 0.0, 1.0};

  cv::Mat rgb_intrinsics =
      cv::Mat(3, 3, CV_64FC1, rgb_intrinsics_data_calibrated);
  cv::Mat rgb_distortion =
      cv::Mat(1, 5, CV_64FC1, rgb_distortion_data_calibrated);
  cv::Mat extrinsics = cv::Mat(4, 4, CV_64FC1, extrinsics_data_calibrated);

  cv::Mat image = cv_image->image;
  cv::Mat image_rect = cv_image_rect->image;
  cv::Mat image_rect_reg = cv_image_rect_reg->image;

  // Build a pointcloud from depth registered image.
  PointSurfelCloudType::Ptr depth_cloud_registered(new PointSurfelCloudType);
  PointSurfelCloudType::Ptr depth_cloud_registered_reshaped(
      new PointSurfelCloudType);
  // projectDepthTo3D(image_rect_reg, rgb_intrinsics, depth_cloud_registered);

  std::vector<cv::Point3f> reshaped_depth_cloud;
  std::vector<cv::Point2f> projected_depth;
  cv::Mat depth_image_registered = cv::Mat::zeros(image.size(), CV_32FC1);
  std::cout << "Hello, marko here2!" << std::endl;
  projectDepthToRGBFrame(image, rgb_intrinsics, depth_intrinsics,
                         rgb_distortion, depth_distortion, extrinsics,
                         &reshaped_depth_cloud, &projected_depth,
                         &depth_image_registered);

  // depth_image_registered.convertTo(depth_image_registered, CV_16UC1, 32767);
  // image_rect_reg.convertTo(image_rect_reg, CV_16UC1, 32767);
  // cv::imwrite(
  //     "/home/panjekm/asl_work/camera_calibration_1306030063/"
  //     "test_depth_reg_modelify.png",
  //     depth_image_registered);
  // cv::imwrite(
  //     "/home/panjekm/asl_work/camera_calibration_1306030063/"
  //     "test_depth_reg.png",
  //     image_rect_reg);

  for (cv::Point3f cv_point : reshaped_depth_cloud) {
    PointSurfelType point;
    point.x = cv_point.x;
    point.y = cv_point.y;
    point.z = cv_point.z;
    if (point.z < 0.0 || std::isnan(point.x)) {
      continue;
    }
    depth_cloud_registered_reshaped->push_back(point);
  }

  cv::Mat depth_cloud(image_rect_reg.size(), CV_32FC3);
  cv::rgbd::depthTo3d(depth_image_registered, rgb_intrinsics, depth_cloud);
  std::cout << "Hello, marko here3!" << std::endl;
  for (size_t u = 0u; u < depth_cloud.rows; ++u) {
    for (size_t v = 0u; v < depth_cloud.cols; ++v) {
      PointSurfelType point;
      point.x = depth_cloud.at<cv::Point3f>(u, v).x;
      point.y = depth_cloud.at<cv::Point3f>(u, v).y;
      point.z = depth_cloud.at<cv::Point3f>(u, v).z;
      if (point.z < 0.0 || std::isnan(point.x)) {
        continue;
      }
      depth_cloud_registered->push_back(point);
    }
  }

  // convert PC2 messages to PCL pointclouds
  PointSurfelCloudType::Ptr depth_cloud_registered_driver(
      new PointSurfelCloudType);
  pcl::PCLPointCloud2 pcl_pointcloud2;
  pcl_conversions::toPCL(*pointcloud_registered_message, pcl_pointcloud2);
  pcl::fromPCLPointCloud2(pcl_pointcloud2, *depth_cloud_registered_driver);

  constexpr double kNumOfNeighbors = 1;
  double ssd = computeCloudToCloudSumOfSquaredDistances(
      depth_cloud_registered_reshaped, depth_cloud_registered_driver,
      kNumOfNeighbors);
  std::cout << "SSD is: " << ssd << std::endl;

  visualize(depth_cloud_registered_reshaped, depth_cloud_registered_driver,
            "depth cloud registered");
  // pcl::io::savePLYFileBinary(
  //     "/home/panjekm/asl_work/camera_calibration_1306030063/"
  //     "depth_cloud.ply",
  //     *transformed_depth_cloud);
  // pcl::io::savePLYFileBinary(
  //     "/home/panjekm/asl_work/camera_calibration_1306030063/"
  //     "depth_cloud_driver.ply",
  //     *depth_cloud_registered_driver);
};

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  ros::Time::init();

  Dataset data;

  rosbag::Bag bag;
  bag.open(
      "/home/panjekm/asl_work/camera_calibration_1306030063/"
      "capture_halfMeter.bag",
      rosbag::bagmode::Read);

  std::vector<std::string> topics;

  const std::string kTopicImage = "/camera/depth/image";
  const std::string kTopicImageRect = "/camera/depth/image_rect";
  const std::string kTopicImageRectReg =
      "/camera/depth_registered/sw_registered/image_rect";
  const std::string kTopicDepthCameraInfo = "/camera/depth/camera_info";
  const std::string kTopicPointcloud = "/camera/depth/points";
  const std::string kTopicPointcloudRegistered =
      "/camera/depth_registered/points";

  topics.push_back(kTopicImage);
  topics.push_back(kTopicImageRect);
  topics.push_back(kTopicImageRectReg);
  topics.push_back(kTopicDepthCameraInfo);
  topics.push_back(kTopicPointcloud);
  topics.push_back(kTopicPointcloudRegistered);

  rosbag::View view(bag, rosbag::TopicQuery(topics));

  BagSubscriber<sensor_msgs::Image> sub_image;
  BagSubscriber<sensor_msgs::Image> sub_image_rect;
  BagSubscriber<sensor_msgs::Image> sub_image_rect_reg;
  BagSubscriber<sensor_msgs::CameraInfo> sub_depth_camera_info;
  BagSubscriber<sensor_msgs::PointCloud2> sub_pointcloud;
  BagSubscriber<sensor_msgs::PointCloud2> sub_pointcloud_registered;

  constexpr size_t kMessageQueueSize = 1000u;

  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image,
                                    sensor_msgs::Image, sensor_msgs::CameraInfo,
                                    sensor_msgs::PointCloud2,
                                    sensor_msgs::PointCloud2>
      sync(sub_image, sub_image_rect, sub_image_rect_reg, sub_depth_camera_info,
           sub_pointcloud, sub_pointcloud_registered, kMessageQueueSize);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, _6, &data));
  for (const rosbag::MessageInstance& message : view) {
    sensor_msgs::Image::ConstPtr image_msg =
        message.instantiate<sensor_msgs::Image>();
    std::string message_topic = message.getTopic();
    if (image_msg.get() != nullptr) {
      if (message_topic.compare(kTopicImage) == 0) {
        sub_image.newMessage(image_msg);
        std::cout << "cool1" << std::endl;
      } else if (message_topic.compare(kTopicImageRect) == 0) {
        sub_image_rect.newMessage(image_msg);
        std::cout << "cool2" << std::endl;
      } else if (message_topic.compare(kTopicImageRectReg) == 0) {
        sub_image_rect_reg.newMessage(image_msg);
        std::cout << "cool3" << std::endl;
      }
    }
    sensor_msgs::CameraInfo::ConstPtr camera_info_msg =
        message.instantiate<sensor_msgs::CameraInfo>();
    if (camera_info_msg.get() != nullptr) {
      if (message_topic.compare(kTopicDepthCameraInfo) == 0) {
        sub_depth_camera_info.newMessage(camera_info_msg);
        std::cout << "cool4" << std::endl;
      }
    }
    sensor_msgs::PointCloud2::ConstPtr pointcloud_msg =
        message.instantiate<sensor_msgs::PointCloud2>();
    if (pointcloud_msg.get() != nullptr) {
      if (message_topic.compare(kTopicPointcloud) == 0) {
        sub_pointcloud.newMessage(pointcloud_msg);
        std::cout << "cool6" << std::endl;
      } else if (message_topic.compare(kTopicPointcloudRegistered) == 0) {
        sub_pointcloud_registered.newMessage(pointcloud_msg);
        std::cout << "cool7" << std::endl;
      }
    }
  }

  bag.close();

  return 1;
}
