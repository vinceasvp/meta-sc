% - 2022.05.25 by Chester.W.Xie - ASVP@SCUT
- [The NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) 是一个乐器识别的数据集。
- 该数据集包含了305,979个样本（musical notes），每一个样本都有唯一的音高、音色以及包络。这些
样本是由1006种乐器产生的。样本时长统一为4秒，采样率也统一在16K Hz; 

➢ 数据集已经划分好三个子集：
- 训练集有 953个类别， 共计289,205个样本。类内最多样本数：440， 最少30， 平均303；
- 验证集有 53个类别， 共计12,678个样本。类内最多样本数：348， 最少83， 平均239；
- 测试集有 53个类别， 共计4096个样本。类内最多样本数：125， 最少22， 平均125；
- 验证集和测试集的类别一致，而训练集与验证集/测试集的类别不重叠。

➢ 对验证集和测试集合并之后的样本数进行统计:
- 拥有大于100个样本的类别数：53
- 拥有大于150个样本的类别数：52
- 拥有大于200个样本的类别数：45
- 拥有大于250个样本的类别数：32
- 拥有大于300个样本的类别数：31

➢ 对训练集的样本进行统计

- 拥有大于450个样本的类别数：0
- 拥有大于440个样本的类别数：377
- 拥有大于400个样本的类别数：382

➢ 根据以上的统计信息，我们可以有以下几种重组数据集的选择：

#### setup 1 (NSynth-100-FS):
将测试集和验证集合并之后，保留拥有大于200个样本的那45类作为新类，然后每类统一保留200个样本。\
接着进一步将每类中的200个样本对半分为训练集和测试集；\
从训练集的382类中抽取55个类别作为基类，每个类别保留400个样本，并进一步划分300样本作为训练集，100个样本作为测试集。


#### setup 2 (NSynth-200-FS):
新类的训练集和测试集与setup 1保持一致.\
从训练集的382类中抽取155个类别作为基类，每个类别保留400个样本，并进一步划分300样本作为训练集，100个样本作为测试集。


#### setup 3 (NSynth-300-FS):
新类的训练集和测试集与setup 1保持一致\
从训练集的382类中抽取255个类别作为基类，每个类别保留400个样本，并进一步划分300样本作为训练集，100个样本作为测试集。


#### setup 4 (NSynth-400-FS):
新类的训练集和测试集与setup 1保持一致\
从训练集的382类中抽取355个类别作为基类，每个类别保留400个样本，并进一步划分300样本作为训练集，100个样本作为测试集。

我们的重组策略：
- 保持原始的音频样本目录结构不变；
- 对原始的meta文件进行相应的筛选后得到对应4种设置的meta文件；
- 根据新的meta文件信息去原始的样本目录中读取相应的数据

数据路径：
```
cd /data/datasets/The_NSynth_Dataset
```
目录结构：
<pre>
dataset_root
├── nsynth-100-fs-meta
│    ├── nsynth-100-fs_train.csv # 包含新旧类的全部训练样本信息，新旧类的区别只是样本数量不同
│    ├── nsynth-100-fs_val.csv  #　包含旧类的全部验证样本信息，注意，只有旧类才有验证数据，新类没有
│    ├── nsynth-100-fs_test.csv　# 包含新旧类的全部测试样本信息，这里新旧类的测试样本数量均一致，即样本均衡
│    └── nsynth-100-fs_vocab.json 　# 存储了100个类别的标签名称
│    
├── nsynth-200-fs-meta
│    ├── nsynth-200-fs_train.csv # 同上
│    ├── nsynth-200-fs_val.csv
│    ├── nsynth-200-fs_test.csv
│    └── nsynth-200-fs_vocab.json
│    
├── nsynth-300-fs-meta
│    ├── nsynth-300-fs_train.csv # 同上
│    ├── nsynth-300-fs_val.csv
│    ├── nsynth-300-fs_test.csv
│    └── nsynth-300-fs_vocab.json
│       
├── nsynth-400-fs-meta
│    ├── nsynth-400-fs_train.csv # 同上
│    ├── nsynth-400-fs_val.csv
│    ├── nsynth-400-fs_test.csv
│    └── nsynth-400-fs_vocab.json
│    
├── nsynth-train　 # Nsynth数据集的原始训练集
│    ├── audio
│    |    ├── bass_acoustic_000-024-025.wav
│    |    └── ....
│    └── examples.json  # 原始的meta信息文件
│
├── nsynth-val  # Nsynth数据集的原始验证集
│    ├── audio
│    |    ├── bass_electronic_018-022-025.wav
│    |    └── ....
│    └── examples.json
│
└── nsynth-test # Nsynth数据集的原始测试集
     ├── audio
     |    ├── bass_electronic_018-022-100.wav
     |    └── ....
     └── examples.json
</pre>

以上的每个csv文件都是统一按照如下的格式：
```
                           filename             instrument instrument_family instrument_source  audio_source
0     guitar_electronic_017-088-075  guitar_electronic_017            guitar        electronic  nsynth-train
1     guitar_electronic_017-088-127  guitar_electronic_017            guitar        electronic  nsynth-train
```
filename加上.wav后缀之后就是完整的样本名，instrument是对应样本的标签，instrument_family和instrument_source不需要用到.\
audio_source代表是该样本存储在原始的哪个文件夹中，因此可以根据以下路径格式读取音频样本：
```
 meta_info = pd.read_csv(...)
 path = os.path.join('/data/datasets/The_NSynth_Dataset', meta_info[audio_source][i], meta_info[filename][i] + '.wav') 
```

 需要特别注意的是，以上的全部重组的meta文件是随机筛选得到的，一般情况下，每次运行筛选程序得到的文件信息均会不一样。\
 因此，为保险起见，请将以上4种设置的meta文件夹复制到自己的工程目录下，音频数据可以不用复制，因为重新下载也会是一样的。

下面我们提供一个初级的脚本来展示如何根据不同的设置来读取数据：\
首先将脚本文件 /data/datasets/The_NSynth_Dataset/load_nsynth_data.py 复制到自己的工程目录下， 然后运行

```
python load_nsynth_data.py --metapath /data/datasets/The_NSynth_Dataset --audiopath /data/datasets/The_NSynth_Dataset --num_class 100 --base_class 55

```
--metapath 可以修改为你保存nsynth-100-fs-meta/nsynth-200-fs-meta ...那4个文件夹的路径
--audiopath 为原始nsynth数据集的保留路径，这个路径是公共的，一般无需改变

对于不同设置的数据集，只需 修改变量 --num_class 100/200/300/400 ，程序就会自动读取相应的meta文件并加载对应的数据.\
同时，需要对应修改 --base_class 55/155/255/355，以对应不同情况基类的类别总数.

程序默认在线读取音频样本并直接转换成为fbank特征，如需要使用其他的时频特征，只需在wave_to_tfr函数中修改即可，\
同时也可以再该函数中加入特征归一化、频谱增强等操作.

# Enjoy the data and code!
