cmake_minimum_required(VERSION 3.5)
project(uuv_mpc_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)




install(PROGRAMS scripts/function_file_raybot_v7.py
 		 scripts/generate_raybot_solver_v7.py 
 		 scripts/run_raybot_solver_v7.py
 		 scripts/acceleration_control.py
 	 DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config 
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*~" 
  EXCLUDE)





ament_package()
