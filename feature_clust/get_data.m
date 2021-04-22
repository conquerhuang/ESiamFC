clear
clc

% �����ļ����洢��·��
feature_root = 'F:\my_research\train_data\Feature_Video';
% ��ø�·���µ������ļ��У���Ƶ���ļ��е���ʽ�洢��
feature_dirs = dir(feature_root);
feature_dirs = feature_dirs(3:end); % ȥ������û�õ�Ŀ¼

% �������ڴ洢��Ƶ����������cell �Լ���Щ�������Լ���ȡ���̻�����Ҫ�õ����м�������
video_features = zeros(1,256*6*6);
video_features_root = string(zeros(2,1));

video_features_index = 1;

f = waitbar(0, 'loading data');
for i = 1:length(feature_dirs)
    % ��ȡ��Ŀ¼�µ������ļ�(��Щ�ļ���Ӧ��Ƶ�����²�ͬĿ���������
    feature_names = dir(fullfile(feature_dirs(i).folder, feature_dirs(i).name, '*.mat'));
    
    % ����Щ���ֽ��б������������Ŀ�����������
    for j = 1:length(feature_names)
        video_features_root(video_features_index) = fullfile(feature_names(j).folder, feature_names(j).name);  % ��ȡ�ļ����ڵ�����Ŀ¼
        
        data = load(video_features_root(video_features_index));                                                % ��ȡ��������
        video_features(video_features_index, :) = reshape(data.data, [1,256*6*6]);

        video_features_index = video_features_index + 1;
    end
    waitbar(i/length(feature_dirs),f,'loading data')
end
close(f)
%ȥ����ȡ������Ϣ�Ĺ����У����ɵ����õ��м����, ֻ�������ھ�������ݡ�
clear i j feature_root feature_names feature_dirs f data video_features_index