% ���ݾ�������index ����ÿһ�����ľ�������

%cluster_centre = zeros(k, size(video_features,2));
cluster_centre = zeros(k, 256,6,6);

for i = 1:k
    % ��ȡ������µ�����
    class_index = result.index==i;
    % �����������������
    class_num = sum(class_index(:));

    % �����������������������������µľ�������
    class_centre = sum(video_features(class_index,:),1)./class_num;

    % ���µ�������ķ�����������б��С�
    cluster_centre(i,:,:,:) = reshape(class_centre,256,6,6);
end

