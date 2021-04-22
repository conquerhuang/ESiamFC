clear
clc

% 特征文件锁存储的路径
feature_root = 'F:\my_research\train_data\Feature_Video';
% 获得该路径下的所有文件夹（视频以文件夹的形式存储）
feature_dirs = dir(feature_root);
feature_dirs = feature_dirs(3:end); % 去除两个没用的目录

% 定义用于存储视频序列特征的cell 以及这些特征，以及读取过程汇总需要用到的中间索引。
video_features = zeros(1,256*6*6);
video_features_root = string(zeros(2,1));

video_features_index = 1;

f = waitbar(0, 'loading data');
for i = 1:length(feature_dirs)
    % 获取该目录下的所有文件(这些文件对应视频序列下不同目标的特征）
    feature_names = dir(fullfile(feature_dirs(i).folder, feature_dirs(i).name, '*.mat'));
    
    % 对这些名字进行遍历，获得所有目标的特征数据
    for j = 1:length(feature_names)
        video_features_root(video_features_index) = fullfile(feature_names(j).folder, feature_names(j).name);  % 获取文件所在的完整目录
        
        data = load(video_features_root(video_features_index));                                                % 获取特征数据
        video_features(video_features_index, :) = reshape(data.data, [1,256*6*6]);

        video_features_index = video_features_index + 1;
    end
    waitbar(i/length(feature_dirs),f,'loading data')
end
close(f)
%去掉获取特征信息的过程中，生成的无用的中间变量, 只保留用于聚类的数据。
clear i j feature_root feature_names feature_dirs f data video_features_index