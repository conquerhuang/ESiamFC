load('cluster_result.mat');
cluster_index = result.index;
cluster_root = result.root;

fid = fopen('result_index.txt','w');
for i = 1: length(cluster_index)
    fprintf(fid, '%d', cluster_index(i));
    fprintf(fid,'\n');
end

fid = fopen('result_root.txt','w');
for i = 1: length(cluster_root)
    fprintf(fid, '%s', cluster_root(i));
    fprintf(fid,'\n');
end