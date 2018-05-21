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
Neg_train_hog = [Train_hog Neg_hog(:, rn(1 : floor(size(negtype, 1)/2)))];
negtraintype = [traintype; negtype(rn(1 : floor(size(negtype, 1)/2)))];
Open_hog = [Test_hog Neg_hog(:, rn(floor(size(negtype, 1)/2) + 1 : end))];
opentype = [testtype; negtype(rn(floor(size(negtype, 1)/2) + 1 : end))];

%% 测试集识别
model = svmtrain(traintype, Train_hog', '-t 0');
[testtype_svm,acc,preb]=svmpredict(zeros(size(testtype, 1), 1),Test_hog', model);

%评估
N = zeros(12, 12);
for i = 1 : 12
    for j = 1 : 12
       N(i, j) = size(find(testtype_svm(find(testtype == j)) == i), 1);
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
model = svmtrain(negtraintype, Neg_train_hog', '-t 0');
[opentype_svm,acc,preb]=svmpredict(zeros(size(opentype, 1), 1),Open_hog', model);

%评估
Nopen = zeros(13, 13);
for i = 1 : 13
    for j = 1 : 13
       Nopen(i, j) = size(find(opentype_svm(find(opentype == j)) == i), 1);
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





