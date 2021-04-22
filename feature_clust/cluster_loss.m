function [loss] = cluster_loss(...
    video_features, cluster_centre, video_features_class, feature_num,k)
% ������ʧ����
% �������룺 
% video_features:���ھ������������������ÿһ�б�ʾһ��������
% cluster_centre:��������У�ÿ����������
% video_features_calss: �������У�ÿ���������������
% feature_num: %���ھ���������ܵ�����
% k:��������У��������

    distance = zeros(feature_num,k);       % �������  �洢�������������ĵľ���
    for i = 1:k
        distance(:,i) = sum(video_features.* cluster_centre(i,:),2); % ���������������������ĵľ���
    end
    distance = -log(0.5 .* (distance + 1));
    loss = 0;
    for i = 1:k %����ÿһ���������
        index = video_features_class == i;
        loss = loss + sum(distance(index,i).^2);
    end
end
