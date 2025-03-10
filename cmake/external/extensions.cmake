# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

message(STATUS "[onnxruntime-extensions] Building onnxruntime-extensions: ${onnxruntime_EXTENSIONS_PATH}")

# add compile definition to enable custom operators in onnxruntime-extensions
add_compile_definitions(ENABLE_EXTENSION_CUSTOM_OPS)

# set options for onnxruntime-extensions
set(OCOS_ENABLE_CTEST OFF CACHE INTERNAL "")
set(OCOS_ENABLE_STATIC_LIB ON CACHE INTERNAL "")
set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "")

# disable exceptions
if (onnxruntime_DISABLE_EXCEPTIONS)
  set(OCOS_ENABLE_CPP_EXCEPTIONS OFF CACHE INTERNAL "")
endif()

# customize operators used
if (onnxruntime_REDUCED_OPS_BUILD)
  set(OCOS_ENABLE_SELECTED_OPLIST ON CACHE INTERNAL "")
endif()

if (onnxruntime_WEBASSEMBLY_DEFAULT_EXTENSION_FLAGS)
  #The generated protobuf files in ORT-extension needs be updated to work with the current protobuf version ORT is using.
  set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_GPT2_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_WORDPIECE_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_BERT_TOKENIZER ON CACHE INTERNAL "")
  set(OCOS_ENABLE_TF_STRING ON CACHE INTERNAL "")
  set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_STATIC_LIB ON CACHE INTERNAL "")
  set(OCOS_ENABLE_BLINGFIRE OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_CV2 OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_OPENCV_CODECS OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_VISION OFF CACHE INTERNAL "")
endif()

# onnxruntime-extensions
if (NOT onnxruntime_EXTENSIONS_OVERRIDDEN)
  onnxruntime_fetchcontent_declare(
    extensions
    URL ${DEP_URL_extensions}
    URL_HASH SHA1=${DEP_SHA1_extensions}
    EXCLUDE_FROM_ALL
  )
  onnxruntime_fetchcontent_makeavailable(extensions)
else()
  # when onnxruntime-extensions is not a subdirectory of onnxruntime,
  # output binary directory must be explicitly specified.
  # and the output binary path is the same as CMake FetchContent pattern
  add_subdirectory(${onnxruntime_EXTENSIONS_PATH} ${CMAKE_BINARY_DIR}/_deps/extensions-subbuild EXCLUDE_FROM_ALL)
endif()

# target library or executable are defined in CMakeLists.txt of onnxruntime-extensions
target_include_directories(ocos_operators PRIVATE ${RE2_INCLUDE_DIR} ${json_SOURCE_DIR}/include)
target_include_directories(ortcustomops PUBLIC ${onnxruntime_EXTENSIONS_PATH}/includes)
if(OCOS_ENABLE_SPM_TOKENIZER)
  onnxruntime_add_include_to_target(sentencepiece-static ${PROTOBUF_LIB} ${ABSEIL_LIBS})
endif()
onnxruntime_add_include_to_target(ocos_operators ${PROTOBUF_LIB} ${ABSEIL_LIBS})
onnxruntime_add_include_to_target(noexcep_operators ${PROTOBUF_LIB} ${ABSEIL_LIBS})

add_dependencies(ocos_operators ${onnxruntime_EXTERNAL_DEPENDENCIES})
add_dependencies(ortcustomops ${onnxruntime_EXTERNAL_DEPENDENCIES})

