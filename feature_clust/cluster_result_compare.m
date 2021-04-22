% load('matlab.mat')
% load('result_compare.mat')

colors = ['r','g','b','c','m','y','k','w'];
k = 6;
selected_iter = 6;

cluster_index_trid = result_compare.index{1}{selected_iter};
cluster_index_bal = result_compare.index{2}{selected_iter};
cluster_num_trid = result_compare.num{1}(selected_iter, :);
cluster_num_bal = result_compare.num{2}(selected_iter, :);

figure
tiledlayout(2,2)

nexttile
for i = 1:k
    class_index = cluster_index_bal == i; %获取当前类别索引
    plot3(features_pca(class_index,1),features_pca(class_index,2),...
        features_pca(class_index,3),[colors(i) '.'])
    if i == 1
        hold on
    end
end
title('balanced','fontsize',24)

nexttile
for i = 1:k
    class_index = cluster_index_trid  == i; %获取当前类别索引
    plot3(features_pca(class_index,1),features_pca(class_index,2),...
        features_pca(class_index,3),[colors(i) '.'])
    if i == 1
        hold on
    end
end
title('triditional', 'fontsize', 24)

nexttile([1, 2])
bar([cluster_num_trid;cluster_num_bal].')
title('number of each cluster', 'fontsize', 24)
legend({'traditional', 'balanced'}, 'fontsize', 24)


