#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>

#include "modelify/alignment_toolbox/alignment_toolbox.h"
#include "modelify/common.h"
#include "modelify/matching_visualizer.h"

using namespace modelify;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  PointSurfelCloudType::Ptr pointcloud_1(new PointSurfelCloudType);
  PointSurfelCloudType::Ptr pointcloud_2(new PointSurfelCloudType);
  std::vector<int> filename_indices =
      pcl::console::parse_file_extension_argument(argc, argv, "ply");
  if (filename_indices.size() == 3) {
    std::string filename = argv[filename_indices[0]];
    if (pcl::io::loadPLYFile(filename, *pointcloud_1) == -1) {
      std::cout << "Was not able to open file " << filename << std::endl;
      return 0;
    }
    filename = argv[filename_indices[1]];
    if (pcl::io::loadPLYFile(filename, *pointcloud_2) == -1) {
      std::cout << "Was not able to open file " << filename << ".\n";
      return 0;
    }
  } else {
    std::cout << "ERROR: no input files provided!" << std::endl;
    std::cout << "INSTRUCTIONS: filename.ply filename2.ply output.ply"
              << std::endl;
    return 0;
  }

  Transformation initial_guess = Eigen::Matrix4f::Identity();
  alignment_toolbox::ICPParams params;
  params.max_correspondence_distance_factor = 100.0;
  params.inlier_distance_threshold_factor = 5.0;
  params.transformation_epsilon = 1e-7;
  params.max_iterations = 1000u;
  const double resolution = std::min(computeCloudResolution(pointcloud_2),
                                     computeCloudResolution(pointcloud_1));

  // Estimate transformation.
  Transformation refined_transform;
  double mean_squared_error;
  double inlier_ratio;
  std::vector<size_t> outlier_indices;

  alignment_toolbox::estimateTransformationPointToPoint<PointSurfelType>(
      pointcloud_1, pointcloud_2, initial_guess, params, resolution,
      &refined_transform, &mean_squared_error, &inlier_ratio, &outlier_indices);
  CHECK_LT(mean_squared_error, 5 * 1e-4);
  CHECK_GT(inlier_ratio, 0.8);

  LOG(INFO) << "ICP with validateAlignment";
  LOG(INFO) << "mse: " << mean_squared_error;
  LOG(INFO) << "inlier_ratio: " << inlier_ratio;
  LOG(INFO) << "outlier_indices.size(): " << outlier_indices.size();
  LOG(INFO) << "refined transformation: " << refined_transform;

  // Appy estimated transformation.
  pcl::transformPointCloud(*(pointcloud_1), *(pointcloud_1), refined_transform);

  // visualize the alignment
  std::unique_ptr<MatchingVisualizer> visualizer;
  const bool kShowWindow = true;
  visualizer.reset(new MatchingVisualizer("Visualizer", kShowWindow));
  visualizer->setPointClouds(pointcloud_1, pointcloud_2);
  visualizer->visualize();

  PointSurfelCloudType::Ptr cloud_merged(new PointSurfelCloudType);
  *cloud_merged += *pointcloud_1;
  *cloud_merged += *pointcloud_2;
  pcl::io::savePLYFileBinary(argv[filename_indices[2]], *cloud_merged);

  return 1;
}
