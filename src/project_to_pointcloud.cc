#include <glog/logging.h>
#include <gtest/gtest.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/io/ply_io.h>

#include <modelify/alignment_toolbox/alignment_toolbox.h>
#include <modelify/common.h>
#include <modelify/pcl_tools.h>

using namespace modelify;

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return 1;
}
