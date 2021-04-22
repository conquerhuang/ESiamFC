% % 研究不同的初始化方法对聚类结果的影响
% % 采用elbow的方法确定k的数量。
% accu_err = zeros(20,4);
% for k = 1:20
%     for simu_time = 1:4
%         data_cluster
%         accu_err(k, simu_time) = cluster_loss(...
%             video_features, cluster_centre, video_features_class_new, feature_num,k);
%     end
% end

% % 研究聚类结果中，类别不均衡问题。
% accu_err = zeros(4,1);
% class_num_sum = zeros(4,6);
% for cluster_time = 1:4
%     data_cluster
%     accu_err(cluster_time) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
%     class_num_sum(cluster_time,:) = class_num';
% end

% % 验证随机数种子的用法
% % 验证成功，初始化一个随机种子以后，可以一直沿用
% s = rng;
% cluster_results = zeros(feature_num, 3);
% for cluster_time = 1:3
%     rng(s)
%     data_cluster
%     cluster_results(:,cluster_time) = video_features_class_new;
% end

% 采用类别数量因子控制每个类别中样本的数量。
% 观察聚类结果中，灭一个类别的样本数量分布情况
% counts = zeros(k, 100);
% centers = zeros(k, 100);
% figure
% for i = 1:k
%     [counts(i,:),centers(i,:)] = hist(distance(:,i),100);
%     subplot(2,3,i)
%     bar(centers(i,:),counts(i,:));
%     title(['cluster ' num2str(i)]);
%     xlabel('distance')
%     ylabel('numbers')
% end

% % 通过因子控制聚类结果中，每一个类别下的样本数量
% %s = rng;
% % 损失控制参数设置 当p=0.005,i=0.005时，会得到收敛结果。
% control_p = 0.005;
% control_i = 0.005;
% 
% rng(s)
% data_cluster_balanced


% 引入吸引子前后，聚类结果损失比较
% 研究聚类结果中，类别不均衡问题。
% accu_err = zeros(10,2);
% class_num_sum = zeros(10,6,2);
% for cluster_time = 1:10
%     s = rng ; % 设置随机数种子
%     % 使用传统方法
%     rng(s)
%     control_p = 0;
%     control_i = 0;
%     data_cluster_balanced
%     accu_err(cluster_time,1) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
%     class_num_sum(cluster_time,:,1) = class_num';
%     % 使用吸引子后
%     rng(s)
%     control_p = 0.005;
%     control_i = 0.005;
%     data_cluster_balanced
%     accu_err(cluster_time,2) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
%     class_num_sum(cluster_time,:,2) = class_num';
% end

% 研究两种聚类方法，收敛过程中，误差的变化。
result_accu_err = cell(40,1);
result_index = cell(40,1);
control_p = 0.005;
control_i = 0.005;
f = waitbar(0,'searching optimizer cluster result');
for cluster_time = 1:40
    data_cluster_balanced
    result_accu_err{cluster_time} = accu_err;
    result_index{cluster_time} = video_features_class_new;
    waitbar(cluster_time/40,f,'searching optimizer cluster result');
end


figure
hold on
for i = 1:40
    plot(result_accu_err{i})
end







