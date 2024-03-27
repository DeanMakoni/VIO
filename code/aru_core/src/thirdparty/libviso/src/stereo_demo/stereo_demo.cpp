/*
Copyright 2012. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

/*
  Documented C++ sample code of stereo visual odometry (modify to your needs)
  To run this demonstration, download the Karlsruhe dataset sequence
  '2010_03_09_drive_0019' from: www.cvlibs.net!
  Usage: ./viso2 path/to/sequence/2010_03_09_drive_0019
*/

#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <chrono>
// #include <png++/png.hpp>
#include <viso_stereo.h>

using namespace std;

int main(int argc, char **argv) {

//  // we need the path name to 2010_03_09_drive_0019 as input argument
//  if (argc < 2) {
//    cerr << "Usage: ./viso2 path/to/sequence/2010_03_09_drive_0019" << endl;
//    return 1;
//  }
//
//  // sequence directory
//  string dir = argv[1];
//
//  // set most important visual odometry parameters
//  // for a full parameter list, look at: viso_stereo.h
//  VisualOdometryStereo::parameters param;
//
//  // calibration parameters for sequence 2010_03_09_drive_0019
//  param.calib.f = 645.24;  // focal length in pixels
//  param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
//  param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
//  param.base = 0.5707;     // baseline in meters
//
//  // init visual odometry
//  Matcher viso(param.match);
//
//  // current pose (this matrix transforms a point from the current
//  // frame's camera coordinates to the first frame's camera coordinates)
//  Matrix pose = Matrix::eye(4);
//
//  // loop through all frames i=0:372
//  for (int32_t i = 1; i < 290; i++) {
//
//    // input file names
//    char base_name_left[256];
//    sprintf(base_name_left, "%d_left.png", i);
//    char base_name_right[256];
//    sprintf(base_name_right, "%d_right.png", i);
//    // string left_img_file_name  = dir + "/I1_" + base_name;
//    // string right_img_file_name = dir + "/I2_" + base_name;
//    string folder1 = "/home/paulamayo/data/husky_data/images/";
//    string folder2 = "/home/paulamayo/data/kitti/2011_09_26"
//                     "/2011_09_26_drive_0039_sync/image_01/data/";
//    string left_img_file_name = folder1 + base_name_left;
//    string right_img_file_name = folder1 + base_name_right;
//    std::cout << left_img_file_name << endl;
//
//    // catch image read/write errors here
//    try {
//
//      // load left and right input image
//      png::image<png::gray_pixel> left_img(left_img_file_name);
//      png::image<png::gray_pixel> right_img(right_img_file_name);
//
//      cv::Mat img_left=cv::imread(left_img_file_name);
//      cv::Mat img_right=cv::imread(right_img_file_name);
//
//      cv::Mat image_1_left_grey, image_1_right_grey; // = image_init.second
//      // .GetImage();
//
//      cv::cvtColor(img_left, image_1_left_grey,
//                   cv::COLOR_BGR2GRAY);
//      cv::cvtColor(img_right, image_1_right_grey,
//                   cv::COLOR_BGR2GRAY);
//
//      // image dimensions
//      int32_t width = image_1_left_grey.cols;
//      int32_t height = image_1_left_grey.rows;
//
//      // convert input images to uint8_t buffer
//      uint8_t *left_img_data =
//          (uint8_t *)malloc(width * height * sizeof(uint8_t));
//      uint8_t *right_img_data =
//          (uint8_t *)malloc(width * height * sizeof(uint8_t));
//      int32_t k = 0;
//      for (int32_t v = 0; v < height; v++) {
//        for (int32_t u = 0; u < width; u++) {
//          left_img_data[k] = left_img.get_pixel(u, v);
//          right_img_data[k] = right_img.get_pixel(u, v);
//          k++;
//        }
//      }
//
//      // status
//      cout << "Processing: Frame: " << i;
//
//      // compute visual odometry
//      int32_t dims[] = {width, height, width};
//      // Perform the estimation
//      auto estimation_start = std::chrono::high_resolution_clock::now();
//
//      viso.pushBack(image_1_left_grey.data, image_1_right_grey.data, dims, false);
//      viso.matchFeatures(2);
//
//      auto estimation_end = std::chrono::high_resolution_clock::now();
//      std::chrono::duration<double> elapsed = estimation_end - estimation_start;
//      cout<< "Computing features takes " << elapsed.count()
//          << " seconds"<<endl;
//      cout << "Adding features takes " << 1 / elapsed.count()
//           << " Hz"<<endl;
//
////      if (viso.process(image_1_left_grey.data, image_1_right_grey.data, dims)) {
////
////        // on success, update current pose
////        pose = pose * Matrix::inv(viso.getMotion());
////        auto estimation_end = std::chrono::high_resolution_clock::now();
////        std::chrono::duration<double> elapsed = estimation_end - estimation_start;
////        cout<< "Computing features takes " << elapsed.count()
////                  << " seconds"<<endl;
////        cout << "Adding features takes " << 1 / elapsed.count()
////                  << " Hz"<<endl;
////
////        // output some statistics
////        double num_matches = viso.getNumberOfMatches();
////        double num_inliers = viso.getNumberOfInliers();
//////        cout << ", Matches: " << num_matches;
//////        cout << ", Inliers: " << 100.0 * num_inliers / num_matches << " %"
//////             << ", Current pose: " << endl;
//////        cout << pose << endl << endl;
////
////      } else {
////        cout << " ... failed!" << endl;
////      }
//
//      // release uint8_t buffers
//      free(left_img_data);
//      free(right_img_data);
//
//      // catch image read errors here
//    } catch (...) {
//      cerr << "ERROR: Couldn't read input files!" << endl;
//      return 1;
//    }
//  }
//
//  // output
//  cout << "Demo complete! Exiting ..." << endl;

  // exit
  return 0;
}
