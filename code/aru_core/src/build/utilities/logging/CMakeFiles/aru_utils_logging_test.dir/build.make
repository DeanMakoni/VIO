# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build

# Include any dependencies generated for this target.
include utilities/logging/CMakeFiles/aru_utils_logging_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include utilities/logging/CMakeFiles/aru_utils_logging_test.dir/compiler_depend.make

# Include the progress variables for this target.
include utilities/logging/CMakeFiles/aru_utils_logging_test.dir/progress.make

# Include the compile flags for this target's objects.
include utilities/logging/CMakeFiles/aru_utils_logging_test.dir/flags.make

utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o: utilities/logging/CMakeFiles/aru_utils_logging_test.dir/flags.make
utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o: ../utilities/logging/tests/aru_utils_logging_test.cpp
utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o: utilities/logging/CMakeFiles/aru_utils_logging_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o -MF CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o.d -o CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o -c /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/logging/tests/aru_utils_logging_test.cpp

utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.i"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/logging/tests/aru_utils_logging_test.cpp > CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.i

utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.s"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/logging/tests/aru_utils_logging_test.cpp -o CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.s

# Object files for target aru_utils_logging_test
aru_utils_logging_test_OBJECTS = \
"CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o"

# External object files for target aru_utils_logging_test
aru_utils_logging_test_EXTERNAL_OBJECTS =

bin/aru_utils_logging_test: utilities/logging/CMakeFiles/aru_utils_logging_test.dir/tests/aru_utils_logging_test.cpp.o
bin/aru_utils_logging_test: utilities/logging/CMakeFiles/aru_utils_logging_test.dir/build.make
bin/aru_utils_logging_test: utilities/logging/libaru_utilities_logging.a
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
bin/aru_utils_logging_test: /usr/local/lib/libglog.so.0.5.0
bin/aru_utils_logging_test: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
bin/aru_utils_logging_test: /usr/local/lib/libprotobuf.so
bin/aru_utils_logging_test: datatypes/libaru_datatypes_pbImage.a
bin/aru_utils_logging_test: datatypes/libaru_datatypes_pbExperience.a
bin/aru_utils_logging_test: datatypes/libaru_datatypes_pbStereoImage.a
bin/aru_utils_logging_test: datatypes/libaru_datatypes_pbTransform.a
bin/aru_utils_logging_test: datatypes/libaru_datatypes_pbLaser.a
bin/aru_utils_logging_test: datatypes/libaru_datatypes_pbIndex.a
bin/aru_utils_logging_test: utilities/logging/CMakeFiles/aru_utils_logging_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/aru_utils_logging_test"
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/aru_utils_logging_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utilities/logging/CMakeFiles/aru_utils_logging_test.dir/build: bin/aru_utils_logging_test
.PHONY : utilities/logging/CMakeFiles/aru_utils_logging_test.dir/build

utilities/logging/CMakeFiles/aru_utils_logging_test.dir/clean:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging && $(CMAKE_COMMAND) -P CMakeFiles/aru_utils_logging_test.dir/cmake_clean.cmake
.PHONY : utilities/logging/CMakeFiles/aru_utils_logging_test.dir/clean

utilities/logging/CMakeFiles/aru_utils_logging_test.dir/depend:
	cd /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/utilities/logging /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging /mnt/c/Users/ndebe/Desktop/Dean/code/aru_core/src/build/utilities/logging/CMakeFiles/aru_utils_logging_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utilities/logging/CMakeFiles/aru_utils_logging_test.dir/depend

