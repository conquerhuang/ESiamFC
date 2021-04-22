% 两个存储了聚类结果的变量，通过result_analysis函数生成 cluster_indexs_trid, cluster_indexs_bal
% 本函数研究聚类结果中，各个簇样本样本数量分布。
cluster_num_tri = zeros(20, 6);
cluster_num_bal = zeros(size(cluster_num_tri));
for i = 1:20
    for j = 1:6
        index = (cluster_indexs_trid{i} == j);
        cluster_num_tri(i, j) = sum(index);
        
        index = (cluster_indexs_bal{i} == j);
        cluster_num_bal(i, j) = sum(index);
    end
end
edges = linspace(0, 5400, 5);
N_tri = histcounts(cluster_num_tri(:),edges);
N_bal = histcounts(cluster_num_bal(:),edges);
figure
bar([N_tri; N_bal].')



