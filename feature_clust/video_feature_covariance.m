% clust_errs_balanced = cell(15,1);
% video_name = 'ILSVRC2015_train_00007001';
video_name = 'ILSVRC2015_train_00001008';

feature_data_dir = ['./video_feature_data' filesep 'feature_data' filesep video_name];
data_dir = ['./video_feature_data' filesep 'data' filesep video_name];

% 读取特征，并计算协方差矩阵
feature_names = dir(feature_data_dir);
feature_names = feature_names(3:end);

video_len = length(feature_names);
features = cell(video_len,1);
for i = 1:video_len
    features{i} = load([feature_names(i).folder filesep feature_names(i).name]);
    features{i}.data = features{i}.data(:) ./ (sum(features{i}.data(:).^2,'all').^(1/2));
end
feature_cov = zeros(video_len);
for i = 1:video_len
    for j = 1:video_len
        feature_cov(i, j) = sum(features{i}.data .* features{j}.data, 'all');
        feature_cov(i, j) =  -log(0.5 .* (feature_cov(i, j) + 1));
    end
end

feature_cov_sum = sum(feature_cov, 2);
feature_cov_sum = mat2gray(feature_cov_sum).* max(feature_cov(:));
[~, max_index] = max(feature_cov_sum);

% 协方差图
figure
surf(feature_cov)
shading interp

hold on
feature_cov_sum = repmat(feature_cov_sum, 1,5);
grid_x = video_len+2:video_len+6;
grid_y = 1:video_len;
[grid_X, grid_Y] = ndgrid(grid_x, grid_y);
surf(grid_X, grid_Y, feature_cov_sum.')
shading interp
xlim([1, video_len + 6])
ylim([1, video_len])
view([0,0,-90])
title('video feature map covariance','fontsize',24)
ylabel('frame','fontsize',24)
xlabel('frame','fontsize',24)

figure
surf(feature_cov_sum)
shading interp


% % clust_errs_balanced = cell(15,1);
% % video_name = 'ILSVRC2015_train_00007001';
% video_name = 'ILSVRC2015_train_00007001';
% 
% 
% feature_data_dir = ['./video_feature_data' filesep 'feature_data' filesep video_name];
% data_dir = ['./video_feature_data' filesep 'data' filesep video_name];
% 
% % 读取特征，并计算协方差矩阵
% feature_names = dir(feature_data_dir);
% feature_names = feature_names(3:end);
% 
% video_len = length(feature_names);
% features = cell(video_len,1);
% for i = 1:video_len
%     features{i} = load([feature_names(i).folder filesep feature_names(i).name]);
%     features{i}.data = features{i}.data(:) ./ (sum(features{i}.data(:).^2,'all').^(1/2));
% end
% feature_cov = zeros(video_len);
% for i = 1:video_len
%     for j = 1:video_len
%         feature_cov(i, j) = sum(features{i}.data .* features{j}.data, 'all');
%         feature_cov(i, j) =  -log(0.5 .* (feature_cov(i, j) + 1));
%     end
% end
% 
% feature_cov_sum = sum(feature_cov, 2);
% feature_cov_sum = mat2gray(feature_cov_sum).* 0.2.*video_len;
% [~, max_index] = max(feature_cov_sum);
% 
% % 协方差图
% figure
% surf(feature_cov)
% shading interp
% 
% hold on
% 
% x = video_len + 5 + feature_cov_sum;
% y = 1:video_len;
% z = max(feature_cov(:)).* ones(size(y));
% plot3(y,x,z,'r-')
% view([0,0,-90])
% ylim([1,video_len+5+0.2*video_len])
% xlim([1,video_len])

