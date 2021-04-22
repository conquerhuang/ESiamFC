% clc
% clear
% 
% load('matlab.mat')
distance_copy = distance;
[~, video_feature_class] = min(distance_copy, [], 2);
for i = 1:size(distance,1)
    distance_copy(i, video_feature_class(i)) = nan;
end
[~, video_feature_class(:,2)]  = min(distance_copy,[],2);

result.index = [video_feature_class(:,1);video_feature_class(:,2)];
result.root = [video_features_root;video_features_root];













