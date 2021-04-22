% 多次随机初始化聚类

%预分配空间用于存储聚类索引结果
cluster_indexs_trid = cell(20,1);
cluster_indexs_bal = cell(20,1);
for iters = 1:20
    s = rng;
    data_cluster
    cluster_indexs_trid{iters} = video_features_class_new;
    rng(s)
    data_cluster_balanced    % 对数据进行聚类
    cluster_indexs_bal{iters} = video_features_class_new; % 存储当前时刻下的聚类结果。
end

% 对这些聚类结果的相似程度进行比较。
JCs_bal = zeros(20);
FMIs_bal = zeros(20);
RIs_bal = zeros(20);

JCs_tri = zeros(20);
FMIs_tri = zeros(20);
RIs_tri = zeros(20);

for i = 1:20
    for j = i :20
        [JC,FMI,RI,~,~,~,~] = cluster_similarty(cluster_indexs_bal{i},cluster_indexs_bal{j});
        JCs_bal(i,j) = JC;
        JCs_bal(j,i) = JC;
        FMIs_bal(i,j) = FMI;
        FMIs_bal(j,i) = FMI;
        RIs_bal(i,j) = RI;
        RIs_bal(j,i) = RI;
        
        [JC,FMI,RI,~,~,~,~] = cluster_similarty(cluster_indexs_trid{i},cluster_indexs_trid{j});
        JCs_tri(i,j) = JC;
        JCs_tri(j,i) = JC;
        FMIs_tri(i,j) = FMI;
        FMIs_tri(j,i) = FMI;
        RIs_tri(i,j) = RI;
        RIs_tri(j,i) = RI;
    end
end

