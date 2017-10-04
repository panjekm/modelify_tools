#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>
#include <ros/ros.h>

#include "modelify/alignment_toolbox/alignment_toolbox.h"
#include "modelify/common.h"
#include "modelify/matching_visualizer.h"
#include "modelify/testing_toolbox/test_fixtures.h"

using namespace modelify;

int main(int argc, char** argv) {
  std::cout << "HELLO MARKO" << std::endl;

  testing_toolbox::TestFixture test_fixture;
  test_fixture.loadYamlConfigFile("test_data/datasets.yaml");
  testing_toolbox::Dataset data;
  test_fixture.loadDataset("toy_robot", &data);

  Transformation initial_guess = Eigen::Matrix4f::Identity();
  alignment_toolbox::ICPParams params;
  params.max_correspondence_distance_factor = 100.0;
  params.inlier_distance_threshold_factor = 5.0;
  params.transformation_epsilon = 1e-7;
  params.max_iterations = 1000u;
  const double resolution =
      std::min(computeCloudResolution(data.point_clouds[1]),
               computeCloudResolution(data.point_clouds[0]));

  // Estimate transformation.
  Transformation refined_transform;
  double mean_squared_error;
  double inlier_ratio;
  std::vector<size_t> outlier_indices;

  alignment_toolbox::estimateTransformationPointToPoint<PointSurfelType>(
      data.point_clouds[0], data.point_clouds[1], initial_guess, params,
      resolution, &refined_transform, &mean_squared_error, &inlier_ratio,
      &outlier_indices);
  CHECK_LT(mean_squared_error, 5 * 1e-4);
  CHECK_GT(inlier_ratio, 0.8);

  LOG(INFO) << "ICP with validateAlignment";
  LOG(INFO) << "mse: " << mean_squared_error;
  LOG(INFO) << "inlier_ratio: " << inlier_ratio;
  LOG(INFO) << "outlier_indices.size(): " << outlier_indices.size();
  LOG(INFO) << "refined transformation: " << refined_transform;

  // Appy estimated transformation.
  pcl::transformPointCloud(*(data.point_clouds[0]), *(data.point_clouds[0]),
                           refined_transform);

  // visualize the alignment
  std::unique_ptr<MatchingVisualizer> visualizer;
  const bool kShowWindow = true;
  visualizer.reset(new MatchingVisualizer("Visualizer", kShowWindow));
  visualizer->setFileName("test_results/aligned_clouds.png");
  visualizer->setPointClouds(data.point_clouds[0], data.point_clouds[1]);
  if (data.keypoints_available) {
    visualizer->setKeyPoints(data.keypoint_clouds[0], data.keypoint_clouds[1]);
  }
  visualizer->visualize();

  PointSurfelCloudType::Ptr cloud_merged(new PointSurfelCloudType);
  *cloud_merged += *data.point_clouds[0];
  *cloud_merged += *data.point_clouds[1];
  pcl::io::savePLYFileBinary("pointclouds_merged.ply", *cloud_merged);

  return 1;
}
