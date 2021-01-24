---
author: 
date: 2021-01-23 15:52+08:00
layout: post
title: "ONNX-TensorRT"
description: ""
mathjax: true
categories:
- 部署
tags:
- TensorRT
- ONNX
- 部署
---

* content
{:toc}
# ONNX-TensorRT

## 第一部分：ONNX-TensorRT工程

[Onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)工程是用来将onnx模型转成tensorrt可用trtmodel的工程，其中包含了解析onnx op的代码，也可以根据需要添加自定义op。

当然如果没有自定义层之类的修改也可以直接使用tensorrt中nvonnxparser.lib解析。

### nvonnxparser库概览

nvonnxparser库的核心代码文件见CMakeLists.txt文件，如下：

```cmake
set(IMPORTER_SOURCES
  NvOnnxParser.cpp
  ModelImporter.cpp
  builtin_op_importers.cpp
  onnx2trt_utils.cpp
  ShapedWeights.cpp
  ShapeTensor.cpp
  LoopHelpers.cpp
  RNNHelpers.cpp
  OnnxAttrs.cpp
)
```

最终，这些代码被编译成动态链接库nvonnxparser.so和静态链接库nvonnxparser_static.a 

```cmake
add_library(nvonnxparser SHARED ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(nvonnxparser PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})
add_library(nvonnxparser_static STATIC ${IMPORTER_SOURCES})
target_include_directories(nvonnxparser_static PUBLIC ${ONNX_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(nvonnxparser_static PUBLIC onnx_proto ${PROTOBUF_LIBRARY} ${TENSORRT_LIBRARY})
```

### 解析流程解读

解析onnx文件流程，包含**createParser**和**parseFromFile**两部分，对应以下两行代码，不熟悉tensorrt解析的可以先简单了解一下再回来看

`nvonnxparser::createParser(*network, gLogger)`

`onnxParser->parseFromFile(source.onnxmodel().c_str(), 1)`

**createParser**是最外层接口，定义在`NvOnnxParser.h`中，返回`IParser`

```c++
/** \brief 创建一个解析器对象
 *
 * \param network 解析器将写入的network
 * \param logger The logger to use
 * \return a new parser object or NULL if an error occurred
 * \see IParser
 */
#ifdef _MSC_VER
TENSORRTAPI IParser* createParser(nvinfer1::INetworkDefinition& network,
                                  nvinfer1::ILogger& logger)
#else
inline IParser* createParser(nvinfer1::INetworkDefinition& network,
                             nvinfer1::ILogger& logger)
#endif
{
    return static_cast<IParser*>(
        createNvOnnxParser_INTERNAL(&network, &logger, NV_ONNX_PARSER_VERSION));
}
```

```c++
/** \class IParser
 *
 * \brief 用于将ONNX模型解析为TensorRT网络定义的对象
 */
class IParser
{
public:
    /** 将序列化的ONNX模型解析到TensorRT网络中。这种方法的诊断价值非常有限。如果由于任何原因（例如不支持的IR版本、不支持的opset等）解析序列化模型失败，则用户有责任拦截并报告错误。到要获得更好的诊断，请使用下面的parseFromFile方法。
     */
    virtual bool parse(void const* serialized_onnx_model,
                       size_t serialized_onnx_model_size,
                       const char* model_path = nullptr)
        = 0;
    
    /** \brief 解析一个onnx模型文件，可以是一个二进制protobuf或者一个文本onnx模型调用里面的Parse方法
     */
    virtual bool parseFromFile(const char* onnxModelFile, int verbosity) = 0;

    /** \brief 检查TensorRT是否支持特定的ONNX模型
     */
    virtual bool supportsModel(void const* serialized_onnx_model,
                               size_t serialized_onnx_model_size,
                               SubGraphCollection_t& sub_graph_collection,
                               const char* model_path = nullptr)
        = 0;

    /** \brief 考虑到用户提供的权重，将序列化的ONNX模型解析到TensorRT网络中
     */
    virtual bool parseWithWeightDescriptors(
        void const* serialized_onnx_model, size_t serialized_onnx_model_size,
        uint32_t weight_count,
        onnxTensorDescriptorV1 const* weight_descriptors)
        = 0;

    /** \brief 返回解析器是否支持指定的运算符
     */
    virtual bool supportsOperator(const char* op_name) const = 0;
//...
    
//...
};

```

nvonnxparser::createParser函数通过`return new onnx2trt::ModelImporter(network, logger)`，返回类ModelImporter，类ModelImporter继承IParser并重写了虚函数，。

```c++
class ModelImporter : public nvonnxparser::IParser
{
protected:
    string_map<NodeImporter> _op_importers;
    virtual Status importModel(::ONNX_NAMESPACE::ModelProto const& model, uint32_t weight_count,
        onnxTensorDescriptorV1 const* weight_descriptors);

private:
    ImporterContext _importer_ctx;
    RefitMap_t mRefitMap;
    std::list<::ONNX_NAMESPACE::ModelProto> _onnx_models; // Needed for ownership of weights
    int _current_node;
    std::vector<Status> _errors;
public:
    ModelImporter(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger)
        : _op_importers(getBuiltinOpImporterMap())
        , _importer_ctx(network, logger, &mRefitMap)
    {
    }
//...
    
//...
}
```

通过`_op_importers(getBuiltinOpImporterMap())`调用`builtin_op_importers.h`中的getBuiltinOpImporterMap()得到所有onnx注册的op，builtin_op_importers中所有的op，都将以DEFINE_BUILTIN_OP_IMPORTER形式出现，只要按照名字和版本注册了，那么当你加载onnx的时候，都会被认识

> builtin_op_importers
>
> - onnxmodel到trtmodel的parse代码。从onnxmodel的input出发，最后，输出trtmodel的输出tensor_ptr；
> - onnx支持的builtin operators包括Conv, Argmax, Unsample,Relu等，具体可以参考operators.md文件；
> - 文件中根据onnx层的类型名调用相应的DEFINE_BUILTIN_OP_IMPORTER(Conv), DEFINE_BUILTIN_OP_IMPORTER(Argmax), DEFINE_BUILTIN_OP_IMPORTER(Unsample), DEFINE_BUILTIN_OP_IMPORTER(Relu)等，从而完成对应层的onnx2trtmodel的parser。

**parseFromFile**解析入口`onnxParser->parseFromFile(source.onnxmodel().c_str(), 1)`，流程如下

调用ModelImporter::parseFromFile开始做解析

然后调用到ModelImporter::parse

然后是ModelImporter::parseWithWeightDescriptors

然后是ModelImporter::importModel

然后是ModelImporter::importInputs，这里ModelImporter::importInput是控制输入的，如果想对onnx的输入尺寸做修改，请修改里面的trt_dims即可

然后是ModelImporter::parseGraph，这里会调用getBuiltinOpImporterMap函数，获得**builtin_op_importers**所有自定义OP

解析时查询op，调用(*importFunc)，跳转到DEFINE_BUILTIN_OP_IMPORTER(op)

```c++
const string_map<NodeImporter>& opImporters = getBuiltinOpImporterMap();
//...

//...
// Dispatch to appropriate converter.
const NodeImporter* importFunc{nullptr};
if (opImporters.count(node.op_type()))
{
    importFunc = &opImporters.at(node.op_type());
}
else
{
    LOG_INFO("No importer registered for op: " << node.op_type() << ". Attempting to import as plugin.");
    importFunc = &opImporters.at("FallbackPluginImporter");
}
std::vector<TensorOrWeights> outputs;

GET_VALUE((*importFunc)(ctx, node, nodeInputs), &outputs);
```

这里importFunc类型是NodeImporter，定义的std::function，输入(ctx, node, nodeInputs)

```c++
typedef std::function<NodeImportResult(
    IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)>
    NodeImporter;
```

DEFINE_BUILTIN_OP_IMPORTER(op)通过宏定义

```c++
#define DECLARE_BUILTIN_OP_IMPORTER(op)                                                                                \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)

#define DEFINE_BUILTIN_OP_IMPORTER(op)                                                                                 \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs);         \
    static const bool op##_registered_builtin_op = registerBuiltinOpImporter(#op, import##op);                         \
    IGNORE_UNUSED_GLOBAL(op##_registered_builtin_op);                                                                  \
    NodeImportResult import##op(                                                                                       \
        IImporterContext* ctx, ::ONNX_NAMESPACE::NodeProto const& node, std::vector<TensorOrWeights>& inputs)
```

主要完成以下三项工作：
1、将onnx输入数据转化为trt要求的数据格式
2、建立trt层，层定义参考**Nvinfer.h**
3、计算trt输出结果

## 第二部分：自定义op流程

TODO：自定义op

### DEFINE_BUILTIN_OP_IMPORTER 

### plugin层定义











https://github.com/NVIDIA/TensorRT/tree/master/plugin



**NvInferRuntimeCommon.h**

IPluginV2：用户实现层的插件类。插件是应用程序实现自定义层的机制。当与IPluginCreator结合使用时，它提供了一种在反序列化期间注册插件和查找插件注册表的机制。

IPluginV2Ext：此接口通过支持不同的输出数据类型和跨批处理的广播，为IPluginV2接口提供了额外的功能

IPluginV2IOExt：此接口通过扩展不同的I/O数据类型和张量格式，为IPluginV2Ext接口提供了额外的功能。

IPluginCreator：用户实现层的插件创建者类。

IPluginRegistry：所有插件的单一注册点，反序列化期间查找插件实现，pluginregistry只支持IPluginV2类型的插件，并且应该有一个相应的IPluginCreator实现。







来源参考

[TensorRT](https://github.com/NVIDIA/TensorRT)

[TensorRT学习（二）通过C++使用](https://blog.csdn.net/yangjf91/article/details/97912773)

[TensorRT学习（三）通过自定义层扩展TensorRT](https://blog.csdn.net/yangjf91/article/details/98184540)

[Onnx-tensorrt详解之nvonnxparser库](https://blog.csdn.net/cxiazaiyu/article/details/94839558)

[实现TensorRT自定义插件(plugin)自由！](https://oldpan.me/archives/tensorrt-plugin-one-post-get-it)

[TensorRT7实现插件流程](https://zhuanlan.zhihu.com/p/296861242)

[TensorRT的自定义算子Plugin的实现](https://blog.csdn.net/u010552731/article/details/106520241)

[TensorRT(4)：NvInferRuntime.h接口头文件分析](https://blog.csdn.net/hjxu2016/article/details/109288673)