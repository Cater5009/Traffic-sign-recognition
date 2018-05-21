close all;
clear;
clc;

maxm = 145;
maxn = 426;
%读入训练集
[trainlist, traintype] = textread('./data/train.txt','%s%n');
Train = zeros(maxm * maxn, size(traintype, 1));
for i = 1 : size(traintype, 1)
    img_path = ['./data/', trainlist{i}];
    img = imread(img_path);
    img = rgb2gray(img);
    img = imresize(img, [maxm, maxn]);
    Train(:, i) = img(1 : maxm * maxn);
end

%读入测试集
[testlist, testtype] = textread('./data/test.txt','%s%n');
Test = zeros(maxm * maxn, size(testtype, 1));
for i = 1 : size(testtype, 1)
    img_path = ['./data/', testlist{i}];
    img = imread(img_path);
    img = rgb2gray(img);
    img = imresize(img, [maxm, maxn]);
    Test(:, i) = img(1 : maxm * maxn);
end

%读入开集
[neglist, negtype] = textread('./data/test_neg.txt','%s%n');
Neg = zeros(maxm * maxn, size(negtype, 1));
for i = 1 : size(negtype, 1)
    img_path = ['./data/', neglist{i}];
    img = imread(img_path);
    img = rgb2gray(img);
    img = imresize(img, [maxm, maxn]);
    Neg(:, i) = img(1 : maxm * maxn);
end
rn = randperm(size(negtype, 1)); 
Neg_train = Neg(:, rn(1 : floor(size(negtype, 1)/2)));
negtraintype = negtype(rn(1 : floor(size(negtype, 1)/2)));
Open = [Test Neg(:, rn(floor(size(negtype, 1)/2) + 1 : end))];
opentype = [testtype; negtype(rn(floor(size(negtype, 1)/2) + 1 : end))];

maxeig = 0;
%差分Train
Train_mean = mean(Train, 2);
for i = 1 : size(traintype, 1)
   Train_c(:, i) = Train(:, i) - Train_mean; 
end
Train_c = double(Train_c);
%差分Test
Test_mean = mean(Test, 2);
for i = 1 : size(testtype, 1)
   Test_c(:, i) = Test(:, i) - Test_mean; 
end
Test_c = double(Test_c);
%差分Neg
Neg_mean = mean(Neg, 2);
for i = 1 : size(negtype, 1)
   Neg_c(:, i) = Neg(:, i) - Neg_mean; 
end
Neg_c = double(Neg_c);
%特征值分解
C = (1/size(traintype, 1)) * (Train_c' * Train_c);
[pre_vec, d] = eig(C);
for i = 1 : size(d, 1)
    pre_d(i) = d(i, i);
end
pre_d = sort(pre_d);
pre_d_sum = 0;
for i = size(pre_d, 2) : -1 : 1
    if pre_d_sum < sum(pre_d) * 0.95
        maxeig = maxeig + 1;
        pre_d_sum = pre_d_sum + pre_d(i);
    else
        break;
    end
end
% pre_d_sum = 0;
% pre_d_sum_r = zeros(size(pre_d, 2), 1);
% for i = size(pre_d, 2) : -1 : 1
%     pre_d_sum = pre_d_sum + pre_d(i);
%     pre_d_sum_r(size(pre_d, 2) - i + 1) = pre_d_sum / sum(pre_d);
% end
% plot(pre_d_sum_r);
% maxeig = 46;
[Vec, D] = eigs(C, maxeig);
Vec = C' * Vec;
for i = 1 : maxeig
   Vec(:, i) = Vec(:, i)/norm(Vec(:, i)); 
end
eigenimg = Train_c * Vec;

% figure;
% for i = 1 : maxeig
%    im = eigenimg(:, i);
%    im = reshape(im, maxm, maxn);
%    subplot(6, 8, i);
%    im = imagesc(im);
%    colormap('gray');
% end
% suptitle('图1-PCA特征图像');

%投影
Train_p = Train_c' * eigenimg;
Test_p = Test_c' * eigenimg;
Neg_p = Neg_c' * eigenimg;
Atest = eigenimg \ Test;
Test_hat = eigenimg * Atest;%重构样本
Atrain = eigenimg \ Train;
Train_hat = eigenimg * Atrain;
Anegtrain = eigenimg \ Neg_train;
Negtrain_hat = eigenimg * Anegtrain;
Aopen = eigenimg \ Open;
Open_hat = eigenimg * Aopen;

%% 测试集识别
testtype_knn = zeros(size(testtype, 1), 1);
dist = zeros(1, size(traintype, 1));
dist_debug = zeros(1, size(testtype, 1));
for i = 1 : size(testtype, 1)
   for j = 1 : size(traintype, 1)
       dist(j) = norm(Atest(:, i) - Atrain(:, j))^2;
   end
   [mindist index] = sort(dist);
   %KNN，K=1
   testtype_knn(i) = traintype(index(1));
   dist_debug(i) = mindist(1);
%    %KNN，K=3
%    preindex = traintype(index(1:3));
%    for j = 1 : 3
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    testtype_knn(i) = preindex(maxindex(1));
%    %KNN，K=5
%    preindex = traintype(index(1:5));
%    for j = 1 : 5
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    testtype_knn(i) = preindex(maxindex(1));
end

%评估
N = zeros(12, 12);
for i = 1 : 12
    for j = 1 : 12
       N(i, j) = size(find(testtype_knn(find(testtype == j)) == i), 1);
    end
end
ptp_i = zeros(12, 1);%每个类别的识别率
pfp_i = zeros(12, 1);%每个类别的虚警率
Nii_sum = 0;
Nij_sum = zeros(12, 1);
for i = 1 : 12
   Nii_sum = Nii_sum + N(i, i);
   ptp_i(i) = N(i, i)/size(find(testtype == i), 1); 
   for j = 1 : 12
      if i ~= j
          Nij_sum(i) = Nij_sum(i) + N(i, j);
      end
   end
   pfp_i(i) = Nij_sum(i)/size(find(testtype ~= i), 1);
end
ptp = Nii_sum/size(testtype, 1);%总体识别率
pfp = sum(Nij_sum)/(11 * size(testtype, 1));

%% 开集识别
negtraintype_knn = zeros(size(negtraintype, 1), 1);
negtraindist = zeros(1, size(traintype, 1));
negtraindist_debug = -1 * ones(1, 12);
for i = 1 : size(negtraintype, 1)
   for j = 1 : size(traintype, 1)
       negtraindist(j) = norm(Anegtrain(:, i) - Atrain(:, j))^2;
   end
   [mindist index] = sort(negtraindist);
   %KNN，K=1   
   negtraindist_debug(i) = mindist(1);
   negtraintype_knn(i) = traintype(index(1));
%    %KNN，K=3
%    preindex = traintype(index(1:3));
%    for j = 1 : 3
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    negtype_knn(i) = preindex(maxindex(1));
%    negdist_debug(i) = mindist(maxindex(1));
%    %KNN，K=5
%    preindex = traintype(index(1:5));
%    for j = 1 : 5
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    negtype_knn(i) = preindex(maxindex(1));
%    negdist_debug(i) = mindist(maxindex(1));
end
th = -1 *ones(1, 12);
for i = 1 : 12
%     temp = max(negtraindist_debug(find(negtraintype_knn == i))); 
%     if size(temp, 2) ~= 0
%         th(i) = temp(1);
%     end
    
    tempmax = max(negtraindist_debug(find(negtraintype_knn == i))); 
    tempmin = min(negtraindist_debug(find(negtraintype_knn == i))); 
    if size(tempmax, 2) ~= 0 && size(tempmin, 2) ~= 0
        th(i) = (tempmax(1) + tempmin(1))/2;
    end
end
opentype_knn = zeros(size(opentype, 1), 1);
dist = zeros(1, size(traintype, 1));
opendist_debug = zeros(1, size(opentype, 1));
for i = 1 : size(opentype, 1)
   for j = 1 : size(traintype, 1)
       dist(j) = norm(Aopen(:, i) - Atrain(:, j))^2;
   end
   [mindist index] = sort(dist);
   %KNN，K=1
   if mindist(1) < th(traintype(index(1))) || th(traintype(index(1))) == -1
       opentype_knn(i) = traintype(index(1));
   else
       opentype_knn(i) = 13;
   end
   opendist_debug(i) = mindist(1);
%    %KNN，K=3
%    preindex = traintype(index(1:3));
%    for j = 1 : 3
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    if mindist(maxindex(1)) < th(preindex(maxindex(1))) || th(preindex(maxindex(1))) == -1
%        opentype_knn(i) = preindex(maxindex(1));
%    else
%        opentype_knn(i) = 13;
%    end
%    %KNN，K=5
%    preindex = traintype(index(1:5));
%    for j = 1 : 5
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    if mindist(maxindex(1)) < th(preindex(maxindex(1))) || th(preindex(maxindex(1))) == -1
%        opentype_knn(i) = preindex(maxindex(1));
%    else
%        opentype_knn(i) = 13;
%    end
end
%评估
Nopen = zeros(13, 13);
for i = 1 : 13
    for j = 1 : 13
       Nopen(i, j) = size(find(opentype_knn(find(opentype == j)) == i), 1);
    end
end
ptp_i_open = zeros(13, 1);%每个类别的识别率
pfp_i_open = zeros(13, 1);%每个类别的虚警率
Nii_sum_open = 0;
Nij_sum_open = zeros(13, 1);
for i = 1 : 13
   Nii_sum_open = Nii_sum_open + Nopen(i, i);
   ptp_i_open(i) = Nopen(i, i)/size(find(opentype == i), 1); 
   for j = 1 : 13
      if i ~= j
          Nij_sum_open(i) = Nij_sum_open(i) + Nopen(i, j);
      end
   end
   pfp_i_open(i) = Nij_sum_open(i)/size(find(opentype ~= i), 1);
end
ptp_open = Nii_sum_open/size(opentype, 1);%总体识别率
pfp_open = sum(Nij_sum_open)/(12 * size(opentype, 1));

