function [JC,FMI,RI,a,b,c,d] = cluster_similarty(index_a,index_b)
%cluster_similarty 用于度量两次聚类结果的相似程度
%   输入参数
%   index_a 聚类索引a
%   index_b聚类索引b
%   输出参数
%   JC,FM,Rand指标。
%   a = |SS|, b=|SD|, c=|DS|, d=|DD|

% 检查输入参数的合法性。
if length(index_a) ~= length(index_b)
    error('illegal input. index_a and index_b with unmatched dim')
end
m = length(index_a);

% 生成a的标记矩阵。
index_mat_a = zeros(m,m);
index_mat_b = index_mat_a;
for i = 1:m
    for j = i:m
        % 为 index_a 生成标记矩阵
        if index_a(i) == index_a(j)
            index_mat_a(i,j) = 1;
            index_mat_a(j,i) = 1;
        else
            index_mat_a(i,j) = 0;
            index_mat_a(j,i) = 0;
        end
        % 为 index_b生成标记矩阵
        if index_b(i) == index_b(j)
            index_mat_b(i,j) = 1;
            index_mat_b(j,i) = 1;
        else
            index_mat_b(i,j) = 0;
            index_mat_b(j,i) = 0;
        end
    end
end
% 根据两个索引矩阵生成 a,b,c,d
a = 0;
b = 0;
c = 0;
d = 0;
for i = 1:m-1
    for j = i+1:m
        if index_mat_a(i,j) == 0 
            if index_mat_b(i,j) == 0 % DD
                d = d+1;
            else                     % DS
                c = c+1;
            end
        else
            if index_mat_b(i,j) == 0 % SD
                b = b+1;
            else                     % SS
                a = a+1;
            end
        end
    end
end
JC = a/(a+b+c);
FMI = sqrt((a/(a+b)) * (a/(a+c)));
RI = 2*(a+d)/(m*(m-1));


end

