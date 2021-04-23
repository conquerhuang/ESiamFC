# ESiamFC
We propose an ensemble Siamese networks (ESiamFC) for tracking. In detail, firestly, we map the train dataset ILSVRC2015 into embedding space. Secondly, we use balanced k-means to cluster th video features. Thirdly, in each cluster we apply transfer learning into SiamFC to obtain k base trackers with their own preferences. Lastly, we propose a Cluster Fusion module which can automatically assign fusion weight to base trackers according th the semantic information of the tracking object.

# Run ESiamFC
## Prepare
### Download model
Download the pretrained model at https://pan.baidu.com/s/10Z0PbuUbnIbwnq3mY2ZNAQ key akhk. Put the pretrained model into "./model" folder. Put video sequence with OTB format into "./video" folder.
### Create enviroment
Create conda enviroment and install the packages listed on the enviroment list "env.txt"
### Run ESiamFC
```bash
demo_siamfc_mix_cw.py
```
Run "demo_siamfc_mix_cw.py" in ".\bin" folder. Impermanent detail and experiments are conducted on the subsequent sections.
## 
# impermanent detail
## train SiamFC
### 1.download dataset
  We use ILSVRC2015 as our training dataset you you can download it at http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz. If you are in chain you can download it at baidu netdisk：https://pan.baidu.com/s/1TTxBRIzFNum3dSF753fIGw  key 6pot.

### 2.create training samples
```bash
create_dataset.py
```
  Run "create_dataset.py" in ".\bin" to crop video sequences. This program crops video sequences into training samples for subsequent training process.

### 3.create lmdb file
```bash
create_lmdb.py
```
  Run "create_lmdb.py" in ".\bin" to create lmdb file In order to read image pairs from hard disk fastly especially for HDD since cv2.imread is too slow.
   (this will take about 50GB storage we strongly recommend you to create this lmdb file on SSD for faster training process)

### 4.train SiamFC
```bash
train_siamfc.py
```
  Run "train_siamfc.py" in ".\bin" to train SiamFC. During the training process the generated model will be stored in ".\models" directory. you can use tensorboard toolbox to observe the train loss and valid loss.

### 5.select model with got10k toolbox
```bash
Siam_FC.py
```
  Run "Siam_FC.py" in "./got10k/trackers" to teset the trained models and select the one with best performance as the baseline for the netx step. The performance of SiamFC trained with Pytorch toolbox on pycharm will be slightly lower than that trained on Matlab (with same hyperparameter).

## generate dataset clusters
If you only intersted in PIk-means. We give out the generated video features on https://pan.baidu.com/s/19in-zFqyOJBnqVwgYmcksA key djdw. You can directly download the generated video features and skip the video feature generate process. jump to step4: "cluster the video features"
### 1.map the dataset into embedding sapce
```bash
generate_feature_dataset.py
```
  Run "generate_feature_dataset.py" in "./bin". this program maps the training dataset ILSVRC2015 into embedding space. It should note that, there may be more thant one target object in one video sequence. We treat these target object on one video sequences as individual samples for next clustering process.

### 2.generate video feature
```bash
video_feature_rate.py
```
  Run "video_feature_rate.py" in "./bin" to generate video feature. We use a covariance matrix to screen these contaminated feature maps with occlusion, motion blur and so on. These contaminated feature maps will significant decrease the reliability of video feature. These contaminated feature maps
  
(note:the after part of this section opearated on matlab!!!)

### 3.acquire video features and use pca to generate 3 dimension video features for debugging.
```bash
get_data.m
feature_pca.m
```
    Run "get_data.m" in "./feature_cluster" to load cluster data to RAM. Run "feature_pca.m" in "./feature_cluster" to get pca data for debug. This process will take about 8GB RAM. best operated on 16GB RAM computer. We strongly recommend you to restorage the generated data to mat file this will shorten the data loading process for the next time. rename the generated "score" matrix to "features_pca". The "features_pca" will be used to visuallize the clustered result on 3D plot. if you are RAM comsume you can only keep the first 3 dim of "features_pca".

### 4.cluster the video features
```bash
data_cluster.m
data_cluster_balanced.m
```
Run "data_cluster.m" in "./feature_cluster" to acquire traditional k-means cluster result. Run "data_cluster_balanced.m" in "./feature_cluster" to acquire PIk-means cluster result. You can use "gpuArray" in matlab to accelerate the cluster process. It will be 10 times faster on GTX1070 GPU than that on E52697v3 (ES) CPU.

### 5.create result for futher process to train base trackers
```bash
create_result.m
```
Run "create_result.m" in "./feature_cluster" to acurire "result_index.txt" and "result_root.txt" files. "result_index.txt" indecate which cluster each video sequence belongs to. "result_root.txt" store the path of each video sequence, which correspond to "result_index.txt"

## Train base trackers
### 1.generate meta data for base trackers.
```
generate_cluster_mata_data.py
```
Run "generate_cluster_mata_data.p"y in "./bin" to generate meta data. The generated meta data will be stored in the training data set folder. these meta data contain indexes of each cluster, which indicate each base tracker's training data set.
    
### 2.train base trackers
```bash
train_siamfc_sep.py
```
Run "train_siamfc_sep.py" in "./bi"n to train each base tracker. Then 6 base trackers will be trained according to the meta data generated in aforementioned process. the base tracker's model will be stored in "./models" folder named "siamfc_cluster*.pth"
    
## Train CW module
### 1.train cw module
```bash
train_siamfc_mix_cw.py
```
Run "train_siamfc_mix_cw.py" in ".\bin" to train the CW module. The trained model will be stored in "./model" named "siamfc_mix_cw.pth". Evaluate the model of each epoch and choose the one with the best performance as the final CW model.

# Experiments
## Impermanent detail
All experiments of the proposed ESiamFC is conducted with the got10k toolbox. for the baseline SiamFC we use the source code developed on pytorch from https://github.com/HonglinChu/SiamTrackers/tree/master/2-SiamFC/SiamFC-VID. The model of SiamFC is downloaded from http://www.robots.ox.ac.uk/~luca/siamese-fc.html which is trained on matlab with matconvnet toolbox. And we transformed the matconvnet model to pytorch model with "convert_pretrained_model.py" in ".\bin" folder. Benchmark results of TColor128 and DTB70 with got10k toolbox is slightly different from the results on matlab with same hyperparameters. We use the SiamFC developed on pytorch as baseline, and take the results of SiamFC on got10k toolbox for comparison

## Evaluate ESiamFC on different banchmark
```bash
SiamFC_mix_cw.py
```
Run "SiamFC_mix_cw.py" in ".\got10k\trackers" to get results of ESiamFC on multiple benchmarks. You can use multiprocessing toolbox to simultaneously run multiple models to save the evaluate time.


