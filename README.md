# NMRP
NMPR模型代码

异质信息网HIN 生成及初始化策略

 1. HINEmbedingNodePreprocess.py  ----->对异质信息网络中节点生成对应的编号文件

 2. genAllMetaPaths.py  -----> 生成异质网络中元路径信息

 3.  利用HIN2vec初始化异质信息网络中节点向量表示, 
 
 4. MetapathBasePathSample.py ---> 对于每种元路径类型, 依据连接点的相似度规则抽样选择相似度最高的前N条路径信息
 
HINDataset.py 表示对HIN数据的读取操作

loaddata_new.py 表示数据集的划分

data_pro_new.py 表示评论信息的预处理操作

NMRPModel 表示整个模型NMRP的计算定义



