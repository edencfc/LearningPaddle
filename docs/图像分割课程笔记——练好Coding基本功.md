# 图像分割课程笔记——练好Coding基本功
本次[图像分割七日打卡营](https://aistudio.baidu.com/aistudio/course/introduce/1767?directly=1&shared=1)确实是如假包换地告别了调包和调参，甚至告别了IDE和代码提示，从最基础的vim开始一行行手输深度学习代码。
当然写作业的时候，还是挺依赖AI Studio上的代码提升，只能算是“半手动”的coding。即使如此，在这个过程中，还是有很多收获的。
## 如何看懂并写出FCN/U-Net/PSPNet/DeepLab代码？
就像代码实战一定是放在课程最后，学习的过程也需要循序渐进。经常调包和调参之后容易养成一些坏习惯，比如要处理某个数据集，先判断以下它的场景，找到对应的模型库，从model zoo里面挑一个指标最好的，然后处理数据、训练、调参。如果只是为了跑一个demo，做一些演示，这种做法可能还行。一旦碰到比赛、或是棘手的数据集，训练指标上不去，不得已需要考虑如何进一步提升的时候，才开始满世界找模型解读，去看原论文——容易看得一头雾水不说，真有哪些tricks要用往往会傻眼，因为不知代码要从何改起。
所以，要读懂代码、改造代码，需要先花时间弄清组网设计过程，切不可本末倒置。
## 逐行敲代码——帮你看懂报错从哪来
弄清原理之后还需要代码实战，coding这个环节大家都知道很重要，但又最容易偷懒。通过这次手敲代码的作业，不仅通过反复回看视频能找回一些coding的感觉，也发现了一些过去感觉莫名其妙的报错从何而来。
比如第4次的deeplab实现，敲代码的时候出现了这些问题：
1. 找不到layers
```python
Traceback (most recent call last):
  File "deeplab.py", line 131, in <module>
    main()
  File "deeplab.py", line 123, in main
    model = DeepLab(num_classes=59)
  File "deeplab.py", line 95, in __init__
    self.layer5 = resnet.layer5
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 533, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'ResNet' object has no attribute 'layer5'
```
所以`layer5`找不到了？回去debug就发现，deeplab一共定义了7层layers，找不到`layer5`说明是`ResNet50()`函数没返回，因此要去排查这个函数。
```python
class DeepLab(Layer):
    def __init__(self,num_classes=59):
        super(DeepLab, self).__init__()
        resnet = ResNet50(pretrained=False)

        self.layer0 = fluid.dygraph.Sequential(
            resnet.conv,
            resnet.pool2d_max
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
       
        # multigrid
        self.layer5 = resnet.layer5
        self.layer6 = resnet.layer6
        self.layer7 = resnet.layer7

        feature_dim = 2048
        self.classifier = DeepLabHead(feature_dim, num_classes)
    
    def forward(self, inputs):
        n, c, h, w = inputs.shape
        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.classifier(x)
        x = fluid.layers.interpolate(x, (h,w), align_corners=False)

        return x
```
然后就在`class ResNet(fluid.dygraph.Layer)`的类定义中找到了这段代码
```python
        if duplicate_blocks:
            self.layer5 = fluid.dygraph.Sequential(
                    *self.make_layer(block,
                                     num_channels[4],
                                     num_filters[3],
                                     depth[3],
                                     stride=1,
                                     name='layer5',
                                     dilation=[x*mgr[0] for x in multi_grid]))
            self.layer6 = fluid.dygraph.Sequential(
                    *self.make_layer(block,
                                     num_channels[4],
                                     num_filters[3],
                                     depth[3],
                                     stride=1,
                                     name='layer6',
                                     dilation=[x*mgr[1] for x in multi_grid]))
            self.layer7 = fluid.dygraph.Sequential(
                    *self.make_layer(block,
                                     num_channels[4],
                                     num_filters[3],
                                     depth[3],
                                     stride=1,
                                     name='layer7',
                                     dilation=[x*mgr[2] for x in multi_grid]))
```
所以，`layer5`是依赖于`if`语句的定义，而`ResNet50()`中`duplicate_blocks`默认`False`
```python
def ResNet50(pretrained=False, duplicate_blocks=False):
    model =  ResNet(layers=50, duplicate_blocks=duplicate_blocks)
    if pretrained:
        model_state, _ = fluid.load_dygraph(model_path['ResNet50'])
        if duplicate_blocks:
            set_dict_ignore_duplicates(model, model_state)
        else:
            model.set_dict(model_state)

    return model
```
这样就好办了，随便改哪一处都行，比如这样：
```python
resnet = ResNet50(pretrained=False, duplicate_blocks=True)
```
2. Notebook项目终止
这个错误就很让人崩溃了，因为报错信息实在太少，而且都是底层报错，没有任何python相关的提示。只能猜测是内存溢出了，可是好好的、一行行对着视频手敲的代码怎么就内存溢出了？
```python
W1025 23:25:26.579102  1152 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
W1025 23:25:26.583094  1152 device_context.cc:260] device: 0, cuDNN Version: 7.6.
W1025 23:25:28.207856  1152 init.cc:226] Warning: PaddlePaddle catches a failure signal, it may not work properly
W1025 23:25:28.207901  1152 init.cc:228] You could check whether you killed PaddlePaddle thread/process accidentally or report the case to PaddlePaddle
W1025 23:25:28.207904  1152 init.cc:231] The detail failure signal is:

W1025 23:25:28.207909  1152 init.cc:234] *** Aborted at 1603639528 (unix time) try "date -d @1603639528" if you are using GNU date ***
W1025 23:25:28.209882  1152 init.cc:234] PC: @                0x0 (unknown)
W1025 23:25:28.210108  1152 init.cc:234] *** SIGSEGV (@0x0) received by PID 1152 (TID 0x7f990d212700) from PID 0; stack trace: ***
W1025 23:25:28.211725  1152 init.cc:234]     @     0x7f990cdfd390 (unknown)
W1025 23:25:28.218042  1152 init.cc:234]     @     0x7f989471a970 paddle::imperative::RuntimeInferVarTypeContext<>::SyncTypeAndDataType()
W1025 23:25:28.221235  1152 init.cc:234]     @     0x7f9895b78aa4 _ZNSt17_Function_handlerIFvPN6paddle9framework19InferVarTypeContextEEZNKS1_7details12OpInfoFillerINS0_9operators18ConvOpInferVarTypeELNS5_14OpInfoFillTypeE3EEclEPKcPNS1_6OpInfoEEUlS3_E_E9_M_invokeERKSt9_Any_dataS3_
W1025 23:25:28.226121  1152 init.cc:234]     @     0x7f98947144bb paddle::imperative::OpBase::Run()
W1025 23:25:28.231685  1152 init.cc:234]     @     0x7f989471c13e paddle::imperative::Tracer::TraceOp()
W1025 23:25:28.237821  1152 init.cc:234]     @     0x7f989471cb08 paddle::imperative::Tracer::TraceOp()
W1025 23:25:28.239933  1152 init.cc:234]     @     0x7f989457472e paddle::pybind::imperative_conv2d()
W1025 23:25:28.241915  1152 init.cc:234]     @     0x7f98945abcea _ZZN8pybind1112cpp_function10initializeIRPFSt10shared_ptrIN6paddle10imperative7VarBaseEERKS6_S8_RKNS_4argsEES6_JS8_S8_SB_EJNS_4nameENS_5scopeENS_7siblingEEEEvOT_PFT0_DpT1_EDpRKT2_ENUlRNS_6detail13function_callEE1_4_FUNESV_
W1025 23:25:28.243752  1152 init.cc:234]     @     0x7f98943b0869 pybind11::cpp_function::dispatcher()
W1025 23:25:28.244269  1152 init.cc:234]     @     0x556afd2a9845 PyCFunction_Call
W1025 23:25:28.244712  1152 init.cc:234]     @     0x556afd345007 _PyEval_EvalFrameDefault
W1025 23:25:28.245226  1152 init.cc:234]     @     0x556afd28956b _PyFunction_FastCallDict
W1025 23:25:28.245769  1152 init.cc:234]     @     0x556afd2a7e53 _PyObject_Call_Prepend
W1025 23:25:28.246331  1152 init.cc:234]     @     0x556afd29adbe PyObject_Call
W1025 23:25:28.246851  1152 init.cc:234]     @     0x556afd341232 _PyEval_EvalFrameDefault
W1025 23:25:28.247253  1152 init.cc:234]     @     0x556afd288539 _PyEval_EvalCodeWithName
W1025 23:25:28.247649  1152 init.cc:234]     @     0x556afd289635 _PyFunction_FastCallDict
W1025 23:25:28.248029  1152 init.cc:234]     @     0x556afd2a7e53 _PyObject_Call_Prepend
W1025 23:25:28.248231  1152 init.cc:234]     @     0x556afd2dfa3a slot_tp_call
W1025 23:25:28.248623  1152 init.cc:234]     @     0x556afd2e08fb _PyObject_FastCallKeywords
W1025 23:25:28.249039  1152 init.cc:234]     @     0x556afd343e86 _PyEval_EvalFrameDefault
W1025 23:25:28.249441  1152 init.cc:234]     @     0x556afd28956b _PyFunction_FastCallDict
W1025 23:25:28.249830  1152 init.cc:234]     @     0x556afd2a7e53 _PyObject_Call_Prepend
W1025 23:25:28.250248  1152 init.cc:234]     @     0x556afd29adbe PyObject_Call
W1025 23:25:28.250675  1152 init.cc:234]     @     0x556afd341232 _PyEval_EvalFrameDefault
W1025 23:25:28.251060  1152 init.cc:234]     @     0x556afd288539 _PyEval_EvalCodeWithName
W1025 23:25:28.251447  1152 init.cc:234]     @     0x556afd289635 _PyFunction_FastCallDict
W1025 23:25:28.251830  1152 init.cc:234]     @     0x556afd2a7e53 _PyObject_Call_Prepend
W1025 23:25:28.252024  1152 init.cc:234]     @     0x556afd2dfa3a slot_tp_call
W1025 23:25:28.252408  1152 init.cc:234]     @     0x556afd2e08fb _PyObject_FastCallKeywords
W1025 23:25:28.252827  1152 init.cc:234]     @     0x556afd3446e8 _PyEval_EvalFrameDefault
W1025 23:25:28.253226  1152 init.cc:234]     @     0x556afd28956b _PyFunction_FastCallDict
Segmentation fault (core dumped)
```
过往的偷懒是要付出代价的——比如因为不理解报错原因，只能一行行和老师的视频比对，最后发现，原来是漏掉了返回值……
```python
class ASPPModule(Layer):
    def __init__(self, num_channels, num_filters, rates):
        super(ASPPModule, self).__init__()
        self.features = []
        self.features.append(
            fluid.dygraph.Sequential(
                Conv2D(num_channels, num_filters, 1),
                BatchNorm(num_filters, act='relu')
            )
        )
        self.features.append(ASPPPooling(num_channels, num_filters))

        for r in rates:
            self.features.append(
                ASPPConv(num_channels, num_filters, r)
            )

        self.project = fluid.dygraph.Sequential(
            Conv2D(num_filters*(2 + len(rates)), num_filters, 1),
            BatchNorm(num_filters, act='relu')
        )

    def forward(self, inputs):
        res = []
        for op in self.features:
            res.append(op(inputs))

        x = fluid.layers.concat(res, axis=1)
        x = self.project(x)

        return x
```
把最后一行`return x`补上，就很顺利地跑通了。收获就是，以后再看到此类报错，第一时间检查每个函数是否有返回值！
```python
W1025 23:31:52.009680  1645 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
W1025 23:31:52.014633  1645 device_context.cc:260] device: 0, cuDNN Version: 7.6.
[2, 59, 512, 512]
```
# 总结
因为时间关系，作业还有很多bonus没有实现，而且很多时候敲还是比较依赖平台的代码提示功能，对一些paddle的API也不太熟悉，本次课程信息量很大，还是需要多回看几遍，才能消化透了。