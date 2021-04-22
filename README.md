# ESiamFC
    We propose an ensemble Siamese networks (ESiamFC) for tracking. In detail, firestly, we map the train dataset ILSVRC2015 into embedding space. Secondly, we use balanced k-means to cluster th video features. Thirdly, in each cluster we apply transfer learning into SiamFC to obtain k base trackers with their own preferences. Lastly, we propose a Cluster Fusion module which can automatically assign fusion weight to base trackers according th the semantic information of the tracking object.

# impermanent detail
## train SiamFC
1.download dataset
  We use ILSVRC2015 as our training dataset you you can download it at http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz. If you are in chain you can download it at baidu netdisk：https://pan.baidu.com/s/1TTxBRIzFNum3dSF753fIGw  key 6pot.

2.create training samples
  run create_dataset.py in .\bin to crop video sequences
  this program is to crop video sequences into training samples for subsequent training process.

3.create lmdb file
  run create_lmdb.py in .\bin to create lmdb file
  In order to faster read image pairs from hard disk especially HDD, it's vital to create lmdb file since cv2.imread is too slow.
   (this will take about 50GB storage we strongly recommend you to create this lmdb file on SSD for faster training process)

4.train SiamFC
  run train_siamfc.py in .\bin to train SiamFC
  During the training process the generated model will be stored in .\models directory. you can use tensorboard toolbox to observe the train loss and valid loss.

5.select model with got10k toolbox
  run Siam_FC.py in ./got10k/trackers to teset the trained models and select the one with best performance as the baseline for the netx step.
  the performance of SiamFC trained with Pytorch toolbox on pycharm will be slightly lower than that trained on Matlab (with same hyperparameter).

## generate dataset clusters

1.map the dataset into embedding sapce
  run generate_feature_dataset.py in ./bin
  this program is to maps the training dataset ILSVRC2015 into embedding space. It should note that, there may be more thant one target object in one video sequence. We treat these target object on one video sequences as individual samples for next clustering process.

2.generate video feature
  run video_feature_rate.py in ./bin to generate video feature.
  we use a covariance matrix to screen these contaminated feature maps with occlusion, motion blur and so on, which will significant decrease the reliability of video features.
  
(note:the after part of this section opearated on matlab!!!)

3.acquire video features and use pca to generate 3 dimension video features for debugging.
    run get_data.m in ./feature_cluster to load cluster data to RAM.
    run feature_pca.m in ./feature_cluster to get pca data for debug.
    This process will take about 8GB RAM. best operated on 16GB RAM computer. We strongly recommend you to restorage the generated data to mat file this will shorten the data loading process for the next time. rename the generated 'score' matrix to features_pca. The features_pca will be used to visuallize the clustered result on 3D plot. if you are RAM comsume you can only keep the first 3 dim of features_pca.

4.cluster the video features
    run data_cluster.m in ./feature_cluster to acquire traditional k-means cluster result.
    run data_cluster_balanced.m in ./feature_cluster to acquire PIk-means cluster result.
    You can use gpuArray in matlab to accelerate the cluster process.  

5.create result for futher process to train base trackers
    run create_result.m in ./feature_cluster to acurire result_index.txt and result_root.txt file.

## Train base trackers
1.generate meta data for base trackers.
    run generate_cluster_mata_data.py in ./bin to generate meta.
    The generated meta data will be stored in the training data set folder where you generate the lmdb file during the "train SiamFC" process. these meta data contain indexes of each cluster which decide each base tracker's training data set.
    
2.train base trackers
    run train_siamfc_sep.py in ./bin to train each base tracker.
    The 6 base trackers will be trained according to the meta data generated in aforementioned process. the base tracker's model will be stored in ./models folder named "siamfc_cluster*.pth"
    
## Train CW module
    train cw module
    run train_siamfc_mix_cw.py in .\bin to train the CW module
    the

# Experiments



