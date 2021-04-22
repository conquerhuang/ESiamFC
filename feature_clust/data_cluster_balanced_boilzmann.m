
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
% ������̣���������Ŷ�����
delta = linspace(1,50,50);
delta = (1/2).^delta;
delta(51:70) = 0;
tao = 10;
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
for iter_time = 1 : 70              % ��ʼ��������

    %----------------------------------------------------------------------
    % ���������������������֮��ľ��롣
    for i = 1:k
        distance(:,i) = sum(video_features.* cluster_centre(i,:),2);
    end
    distance = -log(0.5 .* (distance + 1));
    % distance = exp(-(distance./10000));
    
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
    % �����Ŷ�
    distance = distance + randn(size(distance), 'like', distance).*delta(iter_time);

    %----------------------------------------------------------------------
    % ��ȡ����ÿ�����������һ���������ģ�����������Ϊ�����
    if iter_time == 1
        video_features_class_old = zeros(size(video_features,1),1); % �ϴ˴��������������ͳ�Ƶ�ǰʱ�������ƶ���������
    else
        video_features_class_old = video_features_class_new;
    end
    [~, video_features_class_new] = min(distance, [],2); % �����������һ������Ϊ��ǰ���������

    %----------------------------------------------------------------------
    % �ο��˻��صķ���������ʧ�½������£�����ʧ�����������ʸ���
    % ����ÿ�ε��������еľ�������
    cluster_err = cluster_loss(video_features, cluster_centre, video_features_class_new, feature_num,k);
    if iter_time ~= 1
        if cluster_err < accu_err(iter_time-1) % �������С��ִ�и���
            update_flag = 1;
        else  % ���������ӣ������ʸ���
            disp(['p:' ,num2str(exp(-(cluster_err - accu_err(iter_time-1))/tao))])
            if rand(1) < exp(-(cluster_err - accu_err(iter_time-1))/tao)
                update_flag = 1;
                disp(['energe diff:',num2str(cluster_err - accu_err(iter_time-1))])
                disp('update')
            else
                update_flag = 0;
                disp('not update')
            end
        end
    else
        update_flag = 1;
    end
    %--------------------------------------------------------------------------
    % �����µ��������
    if update_flag
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
        accu_err(iter_time) = cluster_err;
    else
        accu_err(iter_time) = accu_err(iter_time-1);
    end
    



    %----------------------------------------------------------------------
    % ���ݾ������жϵ�ǰ�Ƿ��������������
    class_shift = video_features_class_new == video_features_class_old;
    class_shift_rate = 1 - (sum(class_shift)/length(class_shift));
    if class_shift_rate ==0
        break
    end
    
    for i = 1 : k  % ͳ��ÿ���ط�֧�£�������������
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
    
end


% for i = 1 : k
%     class_index = video_features_class_new == i;
%     class_num(i) = sum(class_index);
% end
% if visualization
%     figure
%     bar(class_num);
% end





