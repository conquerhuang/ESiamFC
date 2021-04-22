
%--------------------------------------------------------------------------
% 聚类参数设置
k = 6;                                 % 聚类类别数目
feature_num = size(video_features, 1); % 用于聚类的样本数量
if exist('control_p', 'var') == 0
    control_p = 0.005;
end
if exist('control_i', 'var') == 0
    control_i = 0.005;
end
distance = zeros(feature_num,k);       % 距离矩阵  存储样本到聚类中心的距离
class_num = floor(ones(k,1).*(feature_num/k)); % 记录迭代过程中，每个类别的样本数量
accu_err = 0;                          % 记录迭代收敛过程中，类别整体误差变化。
%聚类过程可视化相关参数
visualization = 1;
if visualization
    colors = ['r','g','b','c','m','y','k','w'];
    f_cluster_map = figure;
    f_cluster_number = figure;
end
% 用于调试的相关参数
% 用于分析，每一次聚类得到的距离的均值和方差
distance_distribution = zeros(1,2);
distance_ori = nan;
%--------------------------------------------------------------------------
%聚类初始化
% 首先随机初始化k个样本作为k个类别。
% 阶梯式随机采样
a = rand(k,1);
num_stage = floor(linspace(1, feature_num,k+1));
num_stage = num_stage(1:end-1).';
a = floor(a.*(feature_num/k)) + num_stage;

cluster_centre = zeros(k, size(video_features,2));

for i = 1: k
    cluster_centre(i,:) = video_features(a(i),:);
end
%删除部分中间变量
clear a num_stage

%--------------------------------------------------------------------------
% 开始迭代聚类
for iter_time = 1 : 100              % 开始迭代聚类

    %----------------------------------------------------------------------
    % 计算样本与各个聚类中心之间的距离。
    for i = 1:k
        distance(:,i) = sum(video_features.* cluster_centre(i,:),2);
    end
    distance = -log(0.5 .* (distance + 1));
    distance_ori = distance;
    distance_distribution(iter_time, 1) = mean(distance(:));
    distance_distribution(iter_time, 2) = mean(var(distance()));
    
    % 计算每个类别数量偏离均值的程度
    cluster_num_loss_p = (class_num - mean(class_num))/mean(class_num);
    if iter_time == 1
        cluster_num_loss_i = zeros(k,1);
    end
    cluster_num_loss_i = cluster_num_loss_i + cluster_num_loss_p;
    cluster_num_loss = control_p * cluster_num_loss_p...
        + control_i*cluster_num_loss_i;

    
    for i = 1:k
        distance(:,i) = distance(:,i) + cluster_num_loss(i);
    end


    %----------------------------------------------------------------------
    % 获取距离每个样本最近的一个聚类中心，并将样本归为该类别
    if iter_time == 1
        video_features_class_old = zeros(size(video_features,1),1); % 上此次特征的类别，用于统计当前时刻样本移动的数量。
    else
        video_features_class_old = video_features_class_new;
    end
    [~, video_features_class_new] = min(distance, [],2); % 将距离最近的一个类别归为当前样本的类别。


    %--------------------------------------------------------------------------
    % 计算新的类别中心
    for i = 1:k
        % 获取该类别下的索引
        class_index = video_features_class_new==i;
        % 计算该类别的样本数量
        class_num_s = sum(class_index(:));

        % 索引出该类别的所有样本，并计算新的聚类中心
        class_centre = sum(video_features(class_index,:),1)./class_num_s;

        % 将新的类别中心放入类别中心列表中。
        cluster_centre(i, :) = class_centre;
    end


    %----------------------------------------------------------------------
    % 根据聚类结果判断当前是否满足结束条件。
    class_shift = video_features_class_new == video_features_class_old;
    class_shift_rate = 1 - (sum(class_shift)/length(class_shift));
    if class_shift_rate ==0
        break
    end
    
    for i = 1 : k
        class_index = video_features_class_new == i;
        class_num(i) = sum(class_index);
    end

    %----------------------------------------------------------------------
    % 聚类过程可视化代码
    if visualization
        if k<=8
            if rem(iter_time, 1)==0 % 每聚类五次显示一次当前聚类结果。
                figure(f_cluster_map);
                hold off
                for i = 1:k
                    class_index = video_features_class_new == i; %获取当前类别索引
                    plot3(features_pca(class_index,1),features_pca(class_index,2),...
                        features_pca(class_index,3),[colors(i) '.'])
                    if i == 1
                        hold on
                    end
                end
                title(['shift rate:' num2str(class_shift_rate,3) ' iter time ' num2str(iter_time)])
                figure(f_cluster_number);
                bar(class_num);
                pause(0.0000001)
            end
        end

    end
    
    % 计算每次迭代过程中的聚类总误差。
    accu_err(iter_time) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
end


% for i = 1 : k
%     class_index = video_features_class_new == i;
%     class_num(i) = sum(class_index);
% end
% if visualization
%     figure
%     bar(class_num);
% end





