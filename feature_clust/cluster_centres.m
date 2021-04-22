% 根据聚类结果的index 计算每一个类别的聚类中心

%cluster_centre = zeros(k, size(video_features,2));
cluster_centre = zeros(k, 256,6,6);

for i = 1:k
    % 获取该类别下的索引
    class_index = result.index==i;
    % 计算该类别的样本数量
    class_num = sum(class_index(:));

    % 索引出该类别的所有样本，并计算新的聚类中心
    class_centre = sum(video_features(class_index,:),1)./class_num;

    % 将新的类别中心放入类别中心列表中。
    cluster_centre(i,:,:,:) = reshape(class_centre,256,6,6);
end

