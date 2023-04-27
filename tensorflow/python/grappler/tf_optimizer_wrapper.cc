/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "pybind11/pybind11.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

void DetectDevices(
    std::unordered_map<std::string, tensorflow::DeviceProperties>* device_map) {
  tensorflow::SessionOptions options;
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  if (!tensorflow::DeviceFactory::AddDevices(options, "", &devices).ok()) {
    return;
  }

  for (const std::unique_ptr<tensorflow::Device>& device : devices) {
    tensorflow::DeviceProperties& prop = (*device_map)[device->name()];
    prop = tensorflow::grappler::GetDeviceInfo(device->parsed_name());

    // Overwrite the memory limit since users might have requested to use only a
    // fraction of the available device memory.
    const tensorflow::DeviceAttributes& attr = device->attributes();
    prop.set_memory_size(attr.memory_limit());
  }
}
namespace tensorflow {

std::vector<uint8> GraphToFlatGraphDefBuffer(GraphDef def) {
  LOG(ERROR) << "== GraphToFlatGraphDefBuffer: START";
  size_t nodes_buf_size = 0;
  for (const NodeDef& node : def.node()) {
    nodes_buf_size += sizeof(size_t) + node.ByteSizeLong();
  }

  std::vector<uint8> nodes_buf_vec(nodes_buf_size);
  uint8* nodes_buf_it = nodes_buf_vec.data();
  for (const NodeDef& node : def.node()) {
    size_t node_size = node.ByteSizeLong();
    *reinterpret_cast<size_t*>(nodes_buf_it) = node_size;
    nodes_buf_it += sizeof(size_t);
    if (!node.SerializeWithCachedSizesToArray(nodes_buf_it)) {
      LOG(ERROR) << "== GraphToFlatGraphDefBuffer:Unable to serialize NodeDef "
                    "protocol buffer";
      throw std::invalid_argument(
          "Unable to serialize NodeDef protocol buffer");
    }
    nodes_buf_it += node_size;
  }

  def.clear_node();
  size_t graph_info_size = def.ByteSizeLong();
  size_t tf_buf_size = sizeof(size_t) + graph_info_size + nodes_buf_size;

  std::vector<uint8> buf(tf_buf_size);
  uint8* tf_buf = buf.data();

  *reinterpret_cast<size_t*>(tf_buf) = graph_info_size;
  if (!def.SerializeWithCachedSizesToArray(tf_buf + sizeof(size_t))) {
    LOG(ERROR) << "== GraphToFlatGraphDefBuffer: Unable to serialize GraphDef "
                  "protocol buffer";
    throw std::invalid_argument("Unable to serialize GraphDef protocol buffer");
  }
  memcpy(tf_buf + sizeof(size_t) + graph_info_size, nodes_buf_vec.data(),
         nodes_buf_size);
  LOG(ERROR) << "== GraphToFlatGraphDefBuffer: END";
  return buf;
}

Status ParseMetaGraphFromFlatBuffer(py::bytes& flat_graph_def_buffer,
                                    MetaGraphDef& def) {
  LOG(ERROR) << "== ParseMetaGraphFromFlatBuffer: START";
  PyObject* py_buf_ptr = flat_graph_def_buffer.ptr();
  uint8 const* buf =
      reinterpret_cast<uint8 const*>(PyBytes_AsString(py_buf_ptr));
  uint8 const* buf_end = buf + PyBytes_Size(py_buf_ptr);

  size_t meta_graph_info_length = *reinterpret_cast<size_t const*>(buf);
  buf += sizeof(size_t);
  if (!tensorflow::ParseProtoUnlimited(&def, buf, meta_graph_info_length)) {
    LOG(ERROR) << "== ParseMetaGraphFromFlatBuffer: Invalid MetaGraphDef";
    return errors::InvalidArgument("Invalid MetaGraphDef");
  }

  for (buf += meta_graph_info_length; buf < buf_end;) {
    NodeDef node;
    size_t node_length = *reinterpret_cast<size_t const*>(buf);
    buf += sizeof(size_t);
    if (!tensorflow::ParseProtoUnlimited(&node, buf, node_length)) {
      LOG(ERROR) << "== ParseMetaGraphFromFlatBuffer: Invalid NodeDef";
      return errors::InvalidArgument("Invalid NodeDef");
    }
    *def.mutable_graph_def()->add_node() = std::move(node);
    buf += node_length;
  }
  LOG(ERROR) << "== ParseMetaGraphFromFlatBuffer: END";
  return OkStatus();
}

}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_tf_optimizer, m) {
  m.def("TF_OptimizeGraph",
        [](tensorflow::grappler::Cluster* cluster,
           const std::string& serialized_config_proto,
           const std::string& serialized_metagraph, bool verbose,
           const std::string& graph_id,
           bool strip_default_attributes) -> py::bytes {
          std::string out_graph_bytes;
          {
            py::gil_scoped_release gil_release;
            tensorflow::ConfigProto config_proto;
            if (!config_proto.ParseFromString(serialized_config_proto)) {
              throw std::invalid_argument(
                  "The ConfigProto could not be parsed as a valid protocol "
                  "buffer");
            }
            tensorflow::MetaGraphDef metagraph;
            if (!metagraph.ParseFromString(serialized_metagraph)) {
              throw std::invalid_argument(
                  "The MetaGraphDef could not be parsed as a valid protocol "
                  "buffer");
            }

            tensorflow::grappler::ItemConfig item_config;
            // This disables graph optimizations in the older graph optimizer,
            // which tend to overlap / be redundant with those in Grappler.
            item_config.apply_optimizations = false;
            item_config.ignore_user_placement = false;
            std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
                tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                    graph_id, metagraph, item_config);
            if (!grappler_item) {
              throw std::invalid_argument(
                  "Failed to import metagraph, check error log for more info.");
            }

            tensorflow::DeviceBase* cpu_device = nullptr;
            tensorflow::GraphDef out_graph;
            tensorflow::grappler::MetaOptimizer optimizer(cpu_device,
                                                          config_proto);

            MaybeRaiseRegisteredFromStatusWithGIL(
                optimizer.Optimize(cluster, *grappler_item, &out_graph));
            if (strip_default_attributes) {
              tensorflow::StripDefaultAttributes(
                  *tensorflow::OpRegistry::Global(), out_graph.mutable_node());
            }
            if (verbose) {
              optimizer.PrintResult();
            }
            out_graph_bytes = out_graph.SerializeAsString();
          }
          return py::bytes(std::move(out_graph_bytes));
        });

  m.def("TF_OptimizeGraphWithFlatBuffers",
        [](tensorflow::grappler::Cluster* cluster,
           const std::string& serialized_config_proto,
           py::bytes serialized_flat_metagraph, bool verbose,
           const std::string& graph_id,
           bool strip_default_attributes) -> py::bytes {
          tensorflow::GraphDef out_graph;
          {
            py::gil_scoped_release gil_release;
            tensorflow::ConfigProto config_proto;
            if (!config_proto.ParseFromString(serialized_config_proto)) {
              throw std::invalid_argument(
                  "The ConfigProto could not be parsed as a valid protocol "
                  "buffer");
            }
            tensorflow::MetaGraphDef metagraph;
            if (!tensorflow::ParseMetaGraphFromFlatBuffer(
                     serialized_flat_metagraph, metagraph)
                     .ok()) {
              throw std::invalid_argument(
                  "The MetaGraphDef could not be parsed as a valid protocol "
                  "buffer");
            }

            tensorflow::grappler::ItemConfig item_config;
            // This disables graph optimizations in the older graph optimizer,
            // which tend to overlap / be redundant with those in Grappler.
            item_config.apply_optimizations = false;
            item_config.ignore_user_placement = false;
            std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
                tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                    graph_id, metagraph, item_config);
            if (!grappler_item) {
              throw std::invalid_argument(
                  "Failed to import metagraph, check error log for more info.");
            }

            tensorflow::DeviceBase* cpu_device = nullptr;
            tensorflow::grappler::MetaOptimizer optimizer(cpu_device,
                                                          config_proto);

            MaybeRaiseRegisteredFromStatusWithGIL(
                optimizer.Optimize(cluster, *grappler_item, &out_graph));
            if (strip_default_attributes) {
              tensorflow::StripDefaultAttributes(
                  *tensorflow::OpRegistry::Global(), out_graph.mutable_node());
            }
            if (verbose) {
              optimizer.PrintResult();
            }
          }
          std::vector<tensorflow::uint8> buf =
              tensorflow::GraphToFlatGraphDefBuffer(std::move(out_graph));
          printf("!!!!!TF_OptimizeGraphWithFlatBuffers::: BUF SIZE:: %ld\n",
                 buf.size());
          return py::bytes(reinterpret_cast<char*>(buf.data()), buf.size());
        });
}
