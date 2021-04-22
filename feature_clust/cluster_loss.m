function [loss] = cluster_loss(...
    video_features, cluster_centre, video_features_class, feature_num,k)
% 聚类损失函数
% 函数输入： 
% video_features:用于聚类的样本特征向量，每一行表示一个样本。
% cluster_centre:聚类过程中，每个类别的中心
% video_features_calss: 聚类结果中，每个样本所属的类别。
% feature_num: %用于聚类的样本总的数量
% k:聚类过程中，类别数量

    distance = zeros(feature_num,k);       % 距离矩阵  存储样本到聚类中心的距离
    for i = 1:k
        distance(:,i) = sum(video_features.* cluster_centre(i,:),2); % 计算所有样本到聚类中心的距离
    end
    distance = -log(0.5 .* (distance + 1));
    loss = 0;
    for i = 1:k %计算每一个类别的误差
        index = video_features_class == i;
        loss = loss + sum(distance(index,i).^2);
    end
end
