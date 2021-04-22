
%--------------------------------------------------------------------------
% �����������
k = 6;                                 % ���������Ŀ
feature_num = size(video_features, 1); % ���ھ������������
if exist('control_p', 'var') == 0
    control_p = 0.005;
end
if exist('control_i', 'var') == 0
    control_i = 0.005;
end
distance = zeros(feature_num,k);       % �������  �洢�������������ĵľ���
class_num = floor(ones(k,1).*(feature_num/k)); % ��¼���������У�ÿ��������������
accu_err = 0;                          % ��¼�������������У�����������仯��
%������̿��ӻ���ز���
visualization = 1;
if visualization
    colors = ['r','g','b','c','m','y','k','w'];
    f_cluster_map = figure;
    f_cluster_number = figure;
end
% ���ڵ��Ե���ز���
% ���ڷ�����ÿһ�ξ���õ��ľ���ľ�ֵ�ͷ���
distance_distribution = zeros(1,2);
distance_ori = nan;
%--------------------------------------------------------------------------
%�����ʼ��
% ���������ʼ��k��������Ϊk�����
% ����ʽ�������
a = rand(k,1);
num_stage = floor(linspace(1, feature_num,k+1));
num_stage = num_stage(1:end-1).';
a = floor(a.*(feature_num/k)) + num_stage;

cluster_centre = zeros(k, size(video_features,2));

for i = 1: k
    cluster_centre(i,:) = video_features(a(i),:);
end
%ɾ�������м����
clear a num_stage

%--------------------------------------------------------------------------
% ��ʼ��������
for iter_time = 1 : 100              % ��ʼ��������

    %----------------------------------------------------------------------
    % ���������������������֮��ľ��롣
    for i = 1:k
        distance(:,i) = sum(video_features.* cluster_centre(i,:),2);
    end
    distance = -log(0.5 .* (distance + 1));
    distance_ori = distance;
    distance_distribution(iter_time, 1) = mean(distance(:));
    distance_distribution(iter_time, 2) = mean(var(distance()));
    
    % ����ÿ���������ƫ���ֵ�ĳ̶�
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
    % ��ȡ����ÿ�����������һ���������ģ�����������Ϊ�����
    if iter_time == 1
        video_features_class_old = zeros(size(video_features,1),1); % �ϴ˴��������������ͳ�Ƶ�ǰʱ�������ƶ���������
    else
        video_features_class_old = video_features_class_new;
    end
    [~, video_features_class_new] = min(distance, [],2); % �����������һ������Ϊ��ǰ���������


    %--------------------------------------------------------------------------
    % �����µ��������
    for i = 1:k
        % ��ȡ������µ�����
        class_index = video_features_class_new==i;
        % �����������������
        class_num_s = sum(class_index(:));

        % �����������������������������µľ�������
        class_centre = sum(video_features(class_index,:),1)./class_num_s;

        % ���µ�������ķ�����������б��С�
        cluster_centre(i, :) = class_centre;
    end


    %----------------------------------------------------------------------
    % ���ݾ������жϵ�ǰ�Ƿ��������������
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
    % ������̿��ӻ�����
    if visualization
        if k<=8
            if rem(iter_time, 1)==0 % ÿ���������ʾһ�ε�ǰ��������
                figure(f_cluster_map);
                hold off
                for i = 1:k
                    class_index = video_features_class_new == i; %��ȡ��ǰ�������
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
    
    % ����ÿ�ε��������еľ�������
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





