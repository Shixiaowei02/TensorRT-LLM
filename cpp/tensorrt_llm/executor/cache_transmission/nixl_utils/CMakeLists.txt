if(ENABLE_NIXL)
  # Add NIXL configuration
  set(NIXL_ROOT "/opt/nvidia/nvda_nixl")

  # Ensure TARGET_ARCH is defined
  if(NOT DEFINED TARGET_ARCH)
    message(FATAL_ERROR "TARGET_ARCH is not defined")
  endif()

  message(STATUS "Target architecture: ${TARGET_ARCH}")

  # Find NIXL include directory
  find_path(NIXL_INCLUDE_DIR nixl.h HINTS ${NIXL_ROOT}/include)

  # Find all required NIXL libraries
  find_library(NIXL_LIBRARY nixl HINTS ${NIXL_ROOT}/lib/${TARGET_ARCH})
  find_library(NIXL_BUILD_LIBRARY nixl_build
               HINTS ${NIXL_ROOT}/lib/${TARGET_ARCH})
  find_library(SERDES_LIBRARY serdes HINTS ${NIXL_ROOT}/lib/${TARGET_ARCH})
  find_library(UCX_BACKEND_LIBRARY plugin_UCX
               HINTS ${NIXL_ROOT}/lib/${TARGET_ARCH}/plugins)
  find_library(UCX_UTILS_LIBRARY ucx_utils
               HINTS ${NIXL_ROOT}/lib/${TARGET_ARCH})
  find_library(GDS_BACKEND_LIBRARY plugin_GDS
               HINTS ${NIXL_ROOT}/lib/${TARGET_ARCH}/plugins)

  # Display found libraries
  message(STATUS "Found NIXL libraries:")
  message(STATUS "  NIXL: ${NIXL_LIBRARY}")
  message(STATUS "  NIXL Build: ${NIXL_BUILD_LIBRARY}")
  message(STATUS "  SERDES: ${SERDES_LIBRARY}")
  message(STATUS "  UCX Backend: ${UCX_BACKEND_LIBRARY}")
  message(STATUS "  UCX Utils: ${UCX_UTILS_LIBRARY}")
  message(STATUS "  GDS Backend: ${GDS_BACKEND_LIBRARY}")

  # Check if all required libraries were found
  if(NOT NIXL_LIBRARY
     OR NOT NIXL_BUILD_LIBRARY
     OR NOT SERDES_LIBRARY
     OR NOT UCX_BACKEND_LIBRARY
     OR NOT UCX_UTILS_LIBRARY
     OR NOT GDS_BACKEND_LIBRARY)
    message(FATAL_ERROR "One or more NIXL libraries not found")
  endif()

  # Define the NIXL wrapper target
  set(NIXL_WRAPPER_TARGET "tensorrt_llm_nixl_wrapper")

  # Add the NIXL wrapper target
  add_library(${NIXL_WRAPPER_TARGET} SHARED transferAgent.cpp)

  # Add include directories
  target_include_directories(${NIXL_WRAPPER_TARGET} PRIVATE ${NIXL_INCLUDE_DIR})

  # Link against all NIXL libraries
  target_link_libraries(
    ${NIXL_WRAPPER_TARGET}
    PRIVATE ${NIXL_LIBRARY} ${NIXL_BUILD_LIBRARY} ${SERDES_LIBRARY}
            ${UCX_BACKEND_LIBRARY} ${UCX_UTILS_LIBRARY} ${GDS_BACKEND_LIBRARY})
endif()
