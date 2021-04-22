% % �о���ͬ�ĳ�ʼ�������Ծ�������Ӱ��
% % ����elbow�ķ���ȷ��k��������
% accu_err = zeros(20,4);
% for k = 1:20
%     for simu_time = 1:4
%         data_cluster
%         accu_err(k, simu_time) = cluster_loss(...
%             video_features, cluster_centre, video_features_class_new, feature_num,k);
%     end
% end

% % �о��������У���𲻾������⡣
% accu_err = zeros(4,1);
% class_num_sum = zeros(4,6);
% for cluster_time = 1:4
%     data_cluster
%     accu_err(cluster_time) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
%     class_num_sum(cluster_time,:) = class_num';
% end

% % ��֤��������ӵ��÷�
% % ��֤�ɹ�����ʼ��һ����������Ժ󣬿���һֱ����
% s = rng;
% cluster_results = zeros(feature_num, 3);
% for cluster_time = 1:3
%     rng(s)
%     data_cluster
%     cluster_results(:,cluster_time) = video_features_class_new;
% end

% ��������������ӿ���ÿ�������������������
% �۲�������У���һ���������������ֲ����
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

% % ͨ�����ӿ��ƾ������У�ÿһ������µ���������
% %s = rng;
% % ��ʧ���Ʋ������� ��p=0.005,i=0.005ʱ����õ����������
% control_p = 0.005;
% control_i = 0.005;
% 
% rng(s)
% data_cluster_balanced


% ����������ǰ�󣬾�������ʧ�Ƚ�
% �о��������У���𲻾������⡣
% accu_err = zeros(10,2);
% class_num_sum = zeros(10,6,2);
% for cluster_time = 1:10
%     s = rng ; % �������������
%     % ʹ�ô�ͳ����
%     rng(s)
%     control_p = 0;
%     control_i = 0;
%     data_cluster_balanced
%     accu_err(cluster_time,1) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
%     class_num_sum(cluster_time,:,1) = class_num';
%     % ʹ�������Ӻ�
%     rng(s)
%     control_p = 0.005;
%     control_i = 0.005;
%     data_cluster_balanced
%     accu_err(cluster_time,2) = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
%     class_num_sum(cluster_time,:,2) = class_num';
% end

% �о����־��෽�������������У����ı仯��
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







