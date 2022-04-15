# 推荐系统DLRM模型实现
## 调研进度
> * [x]  模型背景，数据，模型，MLCommons提交格式的调研
> * [x]  弄清楚Facebook的数据处理流程
> * [ ]  弄清楚HugeCTR的数据处理流程
> * [x]  弄清楚HugeCTR的两种网络结构
> * [x]  用OneFLow搭建DLRM模型

## 初步调研
### DLRM的提出
- Facebook在2019年7月份提了一个开源深度学习推荐模型DLRM
- https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/
- 模型结构
  ![image](https://user-images.githubusercontent.com/63179313/145521680-30bc721d-93c5-43b2-87a1-53d8d24d807f.png)
- 有Pytorch和Caffe2两个版本，额外还有个Glow C++版本
- 对于MLP部分，使用数据并行加快速度；对于embedding部分，使用模型并行来解决存储问题。在interaction（绿色）这个步骤，他需要多对多的通讯，他提出了一个butterfly shuffle： parts of a minibatch of embedding lookups on all devices.
![image](https://user-images.githubusercontent.com/63179313/145522563-feb1cbe8-d995-4a43-9c43-200472f9bbd1.png)
- 支持数据：Kaggle Display Advertising Challenge Dataset
- 训练结果
  ![image](https://user-images.githubusercontent.com/63179313/145523698-452d0d7d-3211-4aaf-9f8c-0062e8178945.png)
- 代码
    ```
    python版本要大于3.7
    可以对embedding进行模型量化
    有debug_mode，在此模式下，每一次前向计算会打印
    载入训练数据时，需要barrier，end log，barrier,start log，在barrier
    md_flag对应是否使用md_embedding_bag，用于混合维度分配，这里有两个对embedding的trick
    mlp是复制的，使用数据并行。 embedding是分布式的，使用的模型并行。
    优化方法有sgd,rwsadagrad,adagrad三种
    训练过程中记录性能分析，有前向计算和后向计算，训练，测试和打印；也有测试。前向后向分为单机单卡，单机多卡和多机多卡
    有混合精度
    ```
### MLCommons与MLPerf
- https://mlcommons.org/en/
- MLCommons的目标是加速机器学习创新，让每个人都受益。MLCommons以2018年MLPerf benchmark的起步而成立, 通过与50个合作方协作，它迅速发展成为一套衡量机器学习性能和促进机器学习技术透明度的行业指标。
- 2021.12.01 MLCommons发布了MLPerf Training v1.1(机器学习训练性能基准)的新结果。MLPerf Training测量在各种任务中将机器学习模型训练为标准质量目标所需的时间，这些任务包括图像分类、目标检测、自然语言处理、推荐和强化学习。
- 评测指标：MLPerf测量了一个系统的速度：训练某种模型使其达到目标质量所花费的时间。详情可见： https://arxiv.org/pdf/1910.01500.pdf 。下面是对当前benchmark的简短总结：
每个benchmark都定义了使用的数据集，和模型的目标质量：
![image](https://user-images.githubusercontent.com/63179313/145153573-22821e33-d18c-43f3-9569-beadc9365f7f.png)
- 要参与MLPerf的测试，需要遵循 https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc 并且要使用其提供的log接口 https://github.com/mlcommons/logging 提交的results也需要遵循 https://github.com/mlcommons/policies/blob/master/RESULTS_GUIDELINES.md

### MLPerf中对DLRM的测评
1. Azure: https://github.com/mlcommons/training_results_v1.1/tree/main/Azure/benchmarks/dlrm/implementations/hugectr
2. Dell: https://github.com/mlcommons/training_results_v1.1/tree/main/Dell/benchmarks/dlrm/implementations/hugectr
3. GIGABYTE: https://github.com/mlcommons/training_results_v1.1/tree/main/GIGABYTE/benchmarks/dlrm/implementations/merlin_hugectr
4. Inspur: https://github.com/mlcommons/training_results_v1.1/tree/main/Inspur/benchmarks/dlrm/implementations/hugectr
5. Supermicro: https://github.com/mlcommons/training_results_v1.1/tree/main/Supermicro/benchmarks/dlrm/implementations/hugectr
6. NVIDIA: https://github.com/mlcommons/training_results_v1.1/tree/main/NVIDIA/benchmarks/dlrm/implementations/hugectr

他们都基于Nvidia的HugeCTR来做的，不同点粗略看了一下，硬件平台有的不一样，不过用的基本都是DGXA100。可以主要以NVIDIA的实现为参考来看。
HugeCTR的源项目链接：https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples/dlrm
该项目对facebook的 https://github.com/facebookresearch/dlrm 进行了实现

### HugeCTR对DLRM的实现
- 由solver，reader，optimizer, 和model组成。其中solver配置了一些训练有关的参数，reader配置了数据读取有关的参数，optimizer配置了优化方法，model定义了网络结构。
- 其模型结构用json保存，log也存到了同一json中。
- 其编写了一个mlperf_logger工具包，format_ctr_output脚本解析上述的json文件，生成了MLPerf格式的描述性结果，例如 https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_merlin_hugectr/dlrm/result_0.txt 。

- format_ctr_output用到了https://github.com/mlcommons/logging 中的mllog，这是一个MLPerf logging library。这里有其example:https://github.com/mlcommons/logging/blob/master/mlperf_logging/mllog/examples/dummy_example.py


## 数据集
### 网址
https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/
### 介绍
Criteo 1TB Click Logs dataset是criteo给出的用户点击广告的数据，他比kaggle的一个比赛https://www.kaggle.com/c/criteo-display-ad-challenge 的数据更多，为的是给CTR算法提供基准。
### 详情
- 这个数据包含24个文件，每个文件对应一天的数据。
- 每一行是一个广告相关的数据，行是按时间顺序排列的。第一列是label，正样本为点击，负样本为未点击。这里的数据采集使用了随机下采样。
- 有13个整数类型的特征（大多是计数类的特征），和26个类别型特征。类别型特征被哈希到32位上。这些特征的语义被隐藏了，这些特征可能存在缺失值。
- 每一行的格式为：<label> <integer feature 1> … <integer feature 13> <categorical feature 1> … <categorical feature 26>，如果某个值缺失则为空。
- 与kaggle比赛数据集不同的点在于：数据取样与不同的时期，下采样的比率不同，特征的顺序不同并且某些特征的计算方法也发生了改变，类别特征的哈希函数发生了变化。
### 下载
- 下载某一天：http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_XX.gz, where XX goes from 0 to 23.
- 下载全部：curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_{`seq -s ‘,’ 0 23`}.gz





### 
## 用OneFlow搭建dlrm
### 网络结构
#### 论文中给出的结构
![image](https://pic4.zhimg.com/80/v2-b92f64f66033ff52220cc99bba9dd073_1440w.jpg)
![image](https://pic2.zhimg.com/80/v2-aecfd50137885132dc76ab26032ee97d_1440w.jpg)

#### HugeCTR的实现
首先对于dense特征，是(输入)13->512x4->256x4->128x1（输出），这里DenseLayer层是FusedInnerProduct。
对于sparse_id特征，是(输入)26->128x26(输出)，这里的embedding层是用的HybridSparseEmbedding。
然后128x1+128x26（输入）->128x27（输出），这里的DenseLayer层是Interaction。
128x30->1024x4->1024x4->512x4->256x4->1(输出)，这里DenseLayer层是FusedInnerProduct。
（输入）1->1（输出），这里的DenseLayer层是BinaryCrossEntropyLoss
模型结构如下：
![main](https://i.imgur.com/i7UFEle.png)
- FusedInnerProduct
  似乎是全连接层，先将下游网络的几个输出作为输入cat在一起作为全连接层的输入
- Interaction
  原文：This is done by taking the dot product between all pairs of embedding vectors and processed dense features. These dot products are concatenated with the original processed dense features and post-processed with another MLP (the top or output MLP)
  理解的是f31与sparse_embedding1中的每个vector进行dot，然后得到26个“交叉特征”，接着与f31进行cat。

#### Facebook的实现
画了个草图，但是在数据输入那里不是很明白，似乎是将类别特征转化为了“热编码”，且某些类别特征是可以多值的，于是在lookup那里取出来多个vector，对他们进行求sum。
![image](https://user-images.githubusercontent.com/63179313/145544008-10085768-e616-4d8d-9531-9babc08fd646.png)
![image](https://user-images.githubusercontent.com/63179313/145544159-38605445-422c-4bd5-8a32-364ca90b4528.png)

比如输入的dense特征有4个，经过bot mlp，输出为2
embedding有3个，每个sparse_id在lookup后得到的是2维，输出为3x2
将2与3x2连起来得到8，作为top mlp的输入，最后得到1个输出
```
DLRM_Net(
  (emb_l): ModuleList(
    (0): EmbeddingBag(4, 2, mode=sum)
    (1): EmbeddingBag(3, 2, mode=sum)
    (2): EmbeddingBag(2, 2, mode=sum)
  )
  (bot_l): Sequential(
    (0): Linear(in_features=4, out_features=3, bias=True)
    (1): ReLU()
    (2): Linear(in_features=3, out_features=2, bias=True)
    (3): ReLU()
  )
  (top_l): Sequential(
    (0): Linear(in_features=8, out_features=4, bias=True)
    (1): ReLU()
    (2): Linear(in_features=4, out_features=2, bias=True)
    (3): ReLU()
    (4): Linear(in_features=2, out_features=1, bias=True)
    (5): Sigmoid()
  )
)
```

### 并行模式
这里采取的是facebook的并行策略：mlp结构复制到每一张卡，embedding是分布式的，使用的模型并行。然后对于输入数据使用数据并行:

- Embedding做模型并行训练指的是在一个device或者说计算节点上，仅存有一部分Embedding表，HugeCTR中的是15000Mb，每个device进行并行mini batch梯度更新时，仅更新自己节点上的部分Embedding层参数。
- MLP层和interactions进行数据并行训练指的是每个device上已经有了全部模型参数，每个device上利用部分数据计算gradient，再利用allreduce的方法汇总所有梯度进行参数更新。
![image](https://pic4.zhimg.com/80/v2-4028876ef404dea6ecd8af1873e1a2eb_1440w.jpg)


### 混合精度
看到facebook的实现里有混合精度，应该要加上，在oneflowde resnet50代码里面看到有，可以参考一下。


### 准备数据
1. 从此处下载 https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/

2. 克隆facebook的dlrm的项目
```
git clone https://github.com/facebookresearch/dlrm/
cd dlrm
git checkout mlperf
```

3. 安装运行镜像
```
docker build -t dlrm_reference .
docker run -it --rm --network=host --ipc=host --shm-size=1g --ulimit memlock=-1 \
           --ulimit stack=67108864 --gpus=all  -v /data:/data dlrm_reference
```

4. 预处理数据
运行如下脚本。
他会开始先处理数据，然后开始训练，训练过程中脚本会打印Finished training it。这时可以Ctrl+C打断训练。处理后的数据为：day_train.bin和day_test.bin。这个处理数据的过程需要几天+几个TB的存储空间。
```
python dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" \
       --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000 --data-generation=dataset \
       --data-set=terabyte --raw-data-file=/data/day --processed-data-file=/data/day --loss-function=bce \
       --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2048 --print-time \
       --test-freq=102400 --test-mini-batch-size=16384 --test-num-workers=16 --memory-map --mlperf-logging \
       --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle \
       --mlperf-coalesce-sparse-grads --use-gpu
```

5. 将数据转化为ofrecord格式
- 方案一：facebook的dlrm中:dlrm_data_pytorch负责创建data_loader（读取的是bin文件），可以读取kaggle的那个小数据集，也可以读取1TB的那个大数据集，还支持随机生成的数据。我们可以先从小数据集做起。在没有bin文件时，这个文件调用了data_loader_terabyte，在里面将numpy数据转为了bin格式的数据，还使用了data_utils.py，对数据进行了预处理。这些代码可以复用，在其基础上进行修改，可以将raw data以同样的逻辑转为ofrecord格式。
- 方案二：将生成好的day_train.bin和day_test.bin转为ofrecord格式。

### MLCommons的log
#### 下载安装
1. pip install "git+https://github.com/mlperf/logging.git@1.1.0-rc4"
2. cd logging
3. pip install -e .
#### 使用方法
- 主要参考：https://github.com/mlcommons/logging/blob/master/mlperf_logging/mllog/examples/dummy_example.py 根据这个例子，知道流程主要为：
1. from mlperf_logging import mllog
2. mllog.config()
3. mllogger.start()
  mllogger.event()
  mllogger.end()

- 次要参考：facebook的dlrm中mlperf_logger.py
这是一个可以复用的工具包，看起来有如下功能：
1. 对start,event,end进行了再封装，并应用到了分布式上（私有函数_log_print）
2. 对config进行了再封装
3. barrier，没看懂：distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
4. mlperf_submission_log，用于MLPerf提交的log信息生成

- 其他：看起来MLperf中NVIDIA的mlperf_logger中的一些代码可以复用。
使用mlperf_logger这个工具包，
在训练的程序中调用了common，生成了json文件记录训练过程。 之后应该再使用format_ctr_output脚本解析json文件，生成了MLPerf格式的描述性结果，例如https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/results/dgxa100_ngc21.09_merlin_hugectr/dlrm/result_0.txt 。

### 优化器
使用sgd，
学习率
lr = 24.0,
warmup的训练次数
warmup_steps = 2750,
开始衰减
decay_start = 49315,
衰减次数
decay_steps = 27772,
衰减相关的参数
decay_power = 2.0,
衰减到的最小学习率
end_lr = 0.0,

### 测评指标
需要用到早停，
在达到某一精度时停掉，在这期间所花的训练时间就是测评指标



## 进一步调研：facebook dlrm对数据集的处理
### 整体流程
dlrm_s_pytorch.py 在run时，调用dlrm_data_pytorch中的make_criteo_data_and_loaders，来得到train_data, test_data, train_loader, test_loader

train_data与test_data是由CriteoDataset类实现的，下面详细介绍：
1. 首先会处理好一些文件名
```
self.d_path #数据所在的目录'/dataset/criteo_tb/'
self.d_file #数据文件名'day'
self.npzfile #npz文件名 '/dataset/criteo_tb/day'
self.trafile #特征相关 '/dataset/criteo_tb/fea'
```
2. 会根据目标文件名pro_data(terabyte_processed.npz)的存在与否来进行数据准备，如果这个文件存在，直接读取这个预处理的文件pro_data；如果不存在，需要处理原始数据后，再读取处理后的文件。读取时，有一个memory_map的选项（暂时不知道是做什么用的）。这个要读取的文件名用变量file表示，这是一个由用numpy的save_compressed方法生成的npz文件，有如下四个键：
- X_int 是dense_feature
- X_cat 是sparse_feature
- y 是label
- count 是[len1,len2...] count的长度=embedding的个数=sparse_feature的个数，len1表示第一个sparse_feature的取值空间大小。
在读取完数据之后会对数据进行shuffle（根据randomize选项），根据split的值来决定生成train还是val还是test。

3. 需要对原始数据进行处理时，CriteoDataset类的init方法中会调用data_utils中的getCriteoAdData()方法，该方法会返回一个文件名file。西面是对getCriteoAdData()的解读：
- 首先是处理一些文件名
  ```
  self.d_path #数据所在的目录'/dataset/criteo_tb/'
  self.d_file #数据文件名'day'
  self.npzfile #npz文件名 '/dataset/criteo_tb/day'
  self.trafile #特征相关 '/dataset/criteo_tb/fea'
  ```
- 接着统计每个文件的数据量，存储到total_file: '/dataset/criteo_tb/day_day_count.npz'中（如果该文件已存在，略过）。
  total_count，total_per_file是个list，
  对24个gz文件进行循环读取，读取每一个文件的行数，将其记录到total_per_file中对应的位置，并统计出来所有数据量total_count

- 然后是检查24天，对应的'/dataset/criteo_tb/day_0.npz'与'/dataset/criteo_tb/day_0_processed.npz'文件。如有一个不存在，需要对这些文件重新创建。创建npz文件中，有一个dataset_multiprocessing选项，可以使用多线程（没看懂）。单线程处理，对24个文件每个调用process_one_file方法，这个方法会返回下采样后的数据个数，因此需要更新到total_per_file中。


- process_one_file方法：一行行读取，根据sub_sample_rate来进行略过与否（下采样）,每一行经过split('\t')得到如下的list。第0个元素是label，第1-13个元素是dense_feature，存储到到X_int对应的0维，14-40个元素是sparse_feature，将其以16进驻读取转为10进制数字后存储到X_cat对应的0维上。
  ```
  '''
  第0个是label
  1-13是dense feature
  14-40sparse feature
  sparse feature只有一个值
  00:'1'
  01:'5'
  02:'110'
  03:''
  04:'16'
  05:''
  06:'1'
  07:'0'
  08:'14'
  09:'7'
  10:'1'
  11:''
  12:'306'
  13:''
  14:'62770d79'
  15:'e21f5d58'
  16:'afea442f'
  17:'945c7fcf'
  ...
  '''
  ```
  接着count uniques，在单线程里有个字典构成的列表convertDicts，有26个字典，对应sparse_feature编号，key是10进制下的sparse_feature，value是1，表示这个取值存在。
  下面是存储npz文件，文件名为'/dataset/criteo_tb/day_0.npz'，使用np.savez_compressed方法，将X_int，X_cat的转置，还有y存储进去。假如下采样的数据个数为i个，则X_int的shape为(i,13)，X_cat_t的shape为(26,i)，y的shape为（1,i）。
  返回下采样后的数据个数i

- 接下来是记录每个文件下采样个数，即保存total_per_file， total_count根据total_per_file重新计算，total_file为'/dataset/criteo_tb/day_day_count.npz'，他将total_per_file存储了下来。

- 还需要记录map，用于将sparse_feature转为对应的unique_id，文件名为day_fea_dict_0.npz——day_fea_dict_25.npz 里面存储的是一维数组，长度是这个sparse_feature的取值空间大小，例如[5733963 2946788]，表示将5733963 map到0，2946788 map到1， 0 和 1是喂给模型的值。

- 另外需要记录'day_fea_count.npz'，存储的是一个数组，长度为26，表示每个sparse_feature的取值空间大小。

- 在上面的文件准备好后，调用processCriteoAdData重新处理下day_0_processed.npz，目的是将X_cat重新处理，根据convertDicts将里面的原始sparse_id转为unique_id

- 最后调用concatCriteoAdData，将所有day_i.npz文件夹合并起来，生成terabyte_processed.npz。

### 数据样例
我对day_0.gz文件进行了一定的处理，解压后的文件为day_0，存储在14号机器/dataset/criteo_tb/下。我读取了他的前1000行，用上述流程生成了一些npz文件。


## 进一步调研：facebook的dlrm网络结构细节
### 代码
```
def interact_features(self, x, ly):
  if self.arch_interaction_op == "dot":
      # concatenate dense and sparse features
      (batch_size, d) = x.shape
      T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
      # perform a dot product
      Z = torch.bmm(T, torch.transpose(T, 1, 2))
      # append dense feature with the interactions (into a row vector)
      # approach 1: all
      # Zflat = Z.view((batch_size, -1))
      # approach 2: unique
      _, ni, nj = Z.shape
      # approach 1: tril_indices
      # offset = 0 if self.arch_interaction_itself else -1
      # li, lj = torch.tril_indices(ni, nj, offset=offset)
      # approach 2: custom
      offset = 1 if self.arch_interaction_itself else 0
      li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
      lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
      Zflat = Z[:, li, lj]
      # concatenate dense features and interactions
      R = torch.cat([x] + [Zflat], dim=1)
  elif self.arch_interaction_op == "cat":
      # concatenation features (into a row vector)
      R = torch.cat([x] + ly, dim=1)
  else:
      sys.exit(
          "ERROR: --arch-interaction-op="
          + self.arch_interaction_op
          + " is not supported"
      )

  return R
```
### 解读
#### arch_interaction_op为dot
- 输入：假设arch-mlp-bot的结构是13-??-64，则参数x的shape是(b,64),假设arch-sparse-feature-size为64，arch-embedding-size为1000-1000-1000（即有三个sparse_feature），则ly是个长度3的list,每个元素是shape为(b,64)的tensor。

- cat：将x与ly拼接成（b,4,64）的tensor，这类可以理解为有四个以vector表示的feature，其中3个feature来自于sparse，1个feature来自于dense。

- dot: 4个feature之间两两做特征交叉，算上自己与自己交叉，可以构成10个特征，不算自己与自己交叉，可以构成6个特征。这里的实现用的是torch.bmm矩阵乘，得到的结果Z的shape为(b,4,4)，根据self.arch_interaction_itself参数，offset会置为0或1，然后会选取矩阵的下三角区域对应的下标，构成li与lj，例如在(b,4,4的情况下)，offset为1的情况下，li为1，2，2，3，3，3；lj为0，0，1，0，1，2。 表示(1，0),(2,0)...这些矩阵元素为需要的二阶交叉特征。
![image](https://user-images.githubusercontent.com/63179313/146523608-316a5919-1b8c-45a6-9759-497c9fd2aeb4.png)

- cat: 将一阶的dense_feature与二阶的interaction feature进行拼接。Zflat的shape为(b,6)，将其与原来的x cat后是的R的shape为(b,70),将其作为后续mlp-top的输入
#### arch_interaction_op为cat
- cat: 将一阶的dense_feature与一阶的interaction feature进行拼接。得到的R的shape为(b,64+3*64)

## MLPerf v1.1对dlrm的汇总

| **ID** | **Submitter**| **System**| **Processor**| **#**| **Accelerator** | **#** |**Software** |   **Recom-mendation**(1TB Clickthrough+DLRM)|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|1.1-2000|Azure|nd96amsr_a100_v4_ngc21.09_merlin_hugectr	|AMD EPYC 7V12|2|NVIDIA A100-SXM4-80GB (400W)|8|Merlin HugeCTR with NVIDIA DL Frameworks Release 21.09|1.875|
|	1.1-2031|Dell|XE8545x4A100-SXM-80GB	|	AMD EPYC 7713 64-Core Processor|2|NVIDIA A100-SXM4-80GB (500W)|4|	NGC MXNet 21.09 , NGC PyTorch 21.09 , NGC TensorFlow 21.09-tf1|9.522|
|1.1-2035|	GIGABYTE|	GIGABYTE-G492_ID0_A100-SXM-80GBx8|	Intel Xeon Platinum 8362|	2|	NVIDIA A100-SXM-80GB (400W)|	8	|Merlin HugeCTR with NVIDIA DL Frameworks Release 21.09	|						2.098|
|1.1-2048|	Inspur|	NF5488A5|	AMD EPYC 7713 64-Core Processor|	2|	NVIDIA A100-SXM4-80GB (500W)|	8	|hugectr|							1.715|
|1.1-2049|	Inspur|	NF5688M6|	Intel(R) Xeon(R) Platinum 8358|	2	|NVIDIA A100-SXM4-80GB (500W)	|8|	hugectr		|					1.698|
|1.1-2064|	NVIDIA|	dgxa100_ngc21.09_merlin_hugectr|	AMD EPYC 7742|	2|	NVIDIA A100-SXM4-80GB (400W)|	8	|merlin_hugectr NVIDIA Release 21.09		|					1.709|
|1.1-2069|	NVIDIA|	dgxa100_n8_ngc21.09_merlin_hugectr|	AMD EPYC 7742	|16	|NVIDIA A100-SXM4-80GB (400W)|	64	|merlin_hugectr NVIDIA Release 21.09		|					0.685|
|1.1-2073|	NVIDIA|	dgxa100_n14_ngc21.09_merlin_hugectr|	AMD EPYC 7742|	28|	NVIDIA A100-SXM4-80GB (400W)|	112	|merlin_hugectr NVIDIA Release 21.09		|					0.633|
|1.1-2084|	Supermicro|	AS-2124GQ-NART+|	AMD EPYC 7H12|	2|	NVIDIA A100-SXM4-80GB (400W)|	4	|MXNet NVIDIA Release 21.09	|3.363|
|1.1-2085|	Supermicro|	SYS-420GP-TNAR|	Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz|	2	|NVIDIA A100-SXM4-80GB (500W)	|8	|	|2.157|
