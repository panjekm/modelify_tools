#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>

#include <modelify/alignment_toolbox/alignment_toolbox.h>
#include <modelify/common.h>
#include <modelify/pcl_tools.h>

struct FssrType {
  PCL_ADD_POINT4D;   // This adds the members x,y,z which can also be accessed
                     // using the point (which is float[4])
  PCL_ADD_NORMAL4D;  // This adds the member normal[3] which can also be
                     // accessed using the point (which is float[4])
  union {
    struct {
      // RGB union
      union {
        struct {
          uint8_t b;
          uint8_t g;
          uint8_t r;
          uint8_t a;
        };
        float rgb;
        uint32_t rgba;
      };
      float radius;
      float confidence;
      float curvature;
    };
    float data_c[4];
  };
  float value;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    FssrType,  // here we assume a XYZ + "test" (as fields)
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z,
                                   normal_z)(uint32_t, rgba, rgba)(float, value,
                                                                   value))

using namespace modelify;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  pcl::PointCloud<PointSurfelType>::Ptr pointcloud_in(
      new pcl::PointCloud<PointSurfelType>);
  pcl::PointCloud<FssrType>::Ptr pointcloud_out(new pcl::PointCloud<FssrType>);
  std::vector<int> filename_indices =
      pcl::console::parse_file_extension_argument(argc, argv, "ply");
  if (filename_indices.size() == 2) {
    std::string filename = argv[filename_indices[0]];
    if (pcl::io::loadPLYFile(filename, *pointcloud_in) == -1) {
      std::cout << "Was not able to open file " << filename << std::endl;
      return 0;
    }
  } else {
    std::cout << "ERROR: no input files provided!" << std::endl;
    std::cout << "INSTRUCTIONS: filename.ply output.ply" << std::endl;
    return 0;
  }

  // Choose parameters.
  bool kComputeNormals = true;

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*pointcloud_in, *pointcloud_in, indices);
  if (kComputeNormals) {
    constexpr double kNormalRadiusSearch = 0.02;
    computeNormals(pointcloud_in, kNormalRadiusSearch, pointcloud_in);
  }
  pcl::removeNaNNormalsFromPointCloud(*pointcloud_in, *pointcloud_in, indices);

  for (PointSurfelType point : *pointcloud_in) {
    FssrType point_fssr;
    point_fssr.x = point.x;
    point_fssr.y = point.y;
    point_fssr.z = point.z;
    point_fssr.normal_x = point.normal_x;
    point_fssr.normal_y = point.normal_y;
    point_fssr.normal_z = point.normal_z;
    point_fssr.r = point.r;
    point_fssr.g = point.g;
    point_fssr.b = point.b;
    point_fssr.a = point.a;
    point_fssr.value = 1u;

    pointcloud_out->push_back(point_fssr);
  }

  // Export pointcloud.
  pcl::io::savePLYFileBinary(argv[filename_indices[1]], *pointcloud_out);

  return 1;
}
