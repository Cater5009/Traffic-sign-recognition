close all;
clear;
clc;

addpath(genpath('vlfeat'));

maxm = 145;
maxn = 426;
%读入训练集
[trainlist, traintype] = textread('./data/train.txt','%s%n');
Train = zeros(maxm * maxn, size(traintype, 1));
Train_hog = zeros(18*53*31, size(traintype, 1));
for i = 1 : size(traintype, 1)
    img_path = ['./data/', trainlist{i}];
    img = imread(img_path);
    img = rgb2gray(img);    
    img = imresize(img, [maxm, maxn]);
    Train(:, i) = img(1 : maxm * maxn);
    hog = vl_hog(single(img), 8, 'verbose');
    Train_hog(:, i) = hog(1 : 18 * 53 * 31);
end

%读入测试集
[testlist, testtype] = textread('./data/test.txt','%s%n');
Test = zeros(maxm * maxn, size(testtype, 1));
Test_hog = zeros(18*53*31, size(testtype, 1));
for i = 1 : size(testtype, 1)
    img_path = ['./data/', testlist{i}];
    img = imread(img_path);
    img = rgb2gray(img);   
    img = imresize(img, [maxm, maxn]);
    Test(:, i) = img(1 : maxm * maxn);
    hog = vl_hog(single(img), 8, 'verbose');
    Test_hog(:, i) = hog(1 : 18 * 53 * 31);
end

%读入开集
[neglist, negtype] = textread('./data/test_neg.txt','%s%n');
Neg = zeros(maxm * maxn, size(negtype, 1));
Neg_hog = zeros(18*53*31, size(negtype, 1));
for i = 1 : size(negtype, 1)
    img_path = ['./data/', neglist{i}];
    img = imread(img_path);
    img = rgb2gray(img);    
    img = imresize(img, [maxm, maxn]);
    Neg(:, i) = img(1 : maxm * maxn);
    hog = vl_hog(single(img), 8, 'verbose');
    Neg_hog(:, i) = hog(1 : 18 * 53 * 31);
end

rn = randperm(size(negtype, 1)); 
Neg_hog_train = Neg_hog(:, rn(1 : floor(size(negtype, 1)/2)));
negtraintype = negtype(rn(1 : floor(size(negtype, 1)/2)));
Open_hog = [Test_hog Neg_hog(:, rn(floor(size(negtype, 1)/2) + 1 : end))];
opentype = [testtype; negtype(rn(floor(size(negtype, 1)/2) + 1 : end))];

%% 测试集识别
%识别
testtype_knn = zeros(size(testtype, 1), 1);
dist = zeros(1, size(traintype, 1));
dist_debug = zeros(1, size(testtype, 1));
for i = 1 : size(testtype, 1)
   for j = 1 : size(traintype, 1)
       dist(j) = norm(Test_hog(:, i) - Train_hog(:, j))^2;
   end
   [mindist index] = sort(dist);
%    %KNN，K=1
%    testtype_knn(i) = traintype(index(1));
%    dist_debug(i) = mindist(1);
%    %KNN，K=3
%    preindex = traintype(index(1:3));
%    for j = 1 : 3
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    testtype_knn(i) = preindex(maxindex(1));
   %KNN，K=5
   preindex = traintype(index(1:5));
   for j = 1 : 5
       count_preindex(j) = size(find(preindex == preindex(j)), 1);
   end
   maxindex = find(count_preindex == max(count_preindex));
   testtype_knn(i) = preindex(maxindex(1));
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
       negtraindist(j) = norm(Neg_hog_train(:, i) - Train_hog(:, j))^2;
   end
   [mindist index] = sort(negtraindist);
%    %KNN，K=1   
%    negtraindist_debug(i) = mindist(1);
%    negtraintype_knn(i) = traintype(index(1));
%    %KNN，K=3
%    preindex = traintype(index(1:3));
%    for j = 1 : 3
%        count_preindex(j) = size(find(preindex == preindex(j)), 1);
%    end
%    maxindex = find(count_preindex == max(count_preindex));
%    negtype_knn(i) = preindex(maxindex(1));
%    negdist_debug(i) = mindist(maxindex(1));
   %KNN，K=5
   preindex = traintype(index(1:5));
   for j = 1 : 5
       count_preindex(j) = size(find(preindex == preindex(j)), 1);
   end
   maxindex = find(count_preindex == max(count_preindex));
   negtype_knn(i) = preindex(maxindex(1));
   negdist_debug(i) = mindist(maxindex(1));
end
th = -1 *ones(1, 12);
for i = 1 : 12
    temp = max(negtraindist_debug(find(negtraintype_knn == i))); 
    if size(temp, 2) ~= 0
        th(i) = temp(1);
    end
    
%     tempmax = max(negtraindist_debug(find(negtraintype_knn == i))); 
%     tempmin = min(negtraindist_debug(find(negtraintype_knn == i))); 
%     if size(tempmax, 2) ~= 0 && size(tempmin, 2) ~= 0
%         th(i) = (tempmax(1) + tempmin(1))/2;
%     end
end
opentype_knn = zeros(size(opentype, 1), 1);
dist = zeros(1, size(traintype, 1));
opendist_debug = zeros(1, size(opentype, 1));
for i = 1 : size(opentype, 1)
   for j = 1 : size(traintype, 1)
       dist(j) = norm(Open_hog(:, i) - Train_hog(:, j))^2;
   end
   [mindist index] = sort(dist);
%    %KNN，K=1
%    if mindist(1) < th(traintype(index(1))) || th(traintype(index(1))) == -1
%        opentype_knn(i) = traintype(index(1));
%    else
%        opentype_knn(i) = 13;
%    end
%    opendist_debug(i) = mindist(1);
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
   %KNN，K=5
   preindex = traintype(index(1:5));
   for j = 1 : 5
       count_preindex(j) = size(find(preindex == preindex(j)), 1);
   end
   maxindex = find(count_preindex == max(count_preindex));
   if mindist(maxindex(1)) < th(preindex(maxindex(1))) || th(preindex(maxindex(1))) == -1
       opentype_knn(i) = preindex(maxindex(1));
   else
       opentype_knn(i) = 13;
   end
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





