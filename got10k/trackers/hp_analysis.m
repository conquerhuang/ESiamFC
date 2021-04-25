search_param = 3;  % 搜索超参数的位置

load('accum_result.mat')
score = accum_result(:,6).*0.5+accum_result(:,7).*0.4+accum_result(:,8).*0.1;
figure
subplot(121)
plot(accum_result(:,search_param), score,'-r*')

% 加载之前的数据和当前的数据相结合，得到一个新的表格。
% 对表格中的变量进行排序
% 绘制新的表格图
% 存储整合之后的变量。
load('contain.mat')
contain = [contain; accum_result];
contain = sortrows(contain, search_param);
for i = size(contain,1):-1:2
    if contain(i, search_param) == contain(i-1, search_param) 
        contain(i,:) = []; % s删除相同的行
    end
end
subplot(122)
score = contain(:,6).*0.5+contain(:,7).*0.4+contain(:,8).*0.1;
plot(contain(:,search_param), score,'-r*')

save('contain.mat', 'contain')
