%Binary classification for the Breast Cancer dataset using the Bayesian classification function. 
%Done 5-fold cross-validation
%The data is downloaded from the UCI Machine Learning Repository:
%https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

clear all; close all; clc
load BreastCancer.csv;
class= [1,2];
Y =[class]; 
X = BreastCancer (:,1:9);
Y = BreastCancer (:,10);
totalSamples=116;
%-----------------------------------------------------------------------------------
%assign binary values 0 and 1
for a=1:116
    if Y(a)<2
        Y(a)=0;%For healthy ppl
    else
        Y(a)=1;%for patients
    end
end

N=5;
accuracyMatrix = []; confusionMatrix = [];
% 5 fold 
%Divide the data into 5 sets in each class
class = unique(Y);
for a = 1:size(class,1)
    NoOfFolds = [];
    temp = X(Y==class(a),:);
    index = find(Y==class(a));
    c = numel(index);
    %Shuffle the dataset randomly.
    rng('shuffle');
    temp = temp(randperm(c),:);
    quo = floor(c/N);
    rem = mod(c, N);
    rng('shuffle');
    extra = randsample(1:N,rem);
    NoOfFolds = ones(N,1) * quo;
    NoOfFolds(extra) = NoOfFolds(extra) + 1;
    %Cumulative Sum
    foldSizeCumSum = cumsum(NoOfFolds);
    %Split the dataset into 5 groups
    for j = 1:N
        if j==1
            eval(['nfold.class' , num2str(a) , '.fold', num2str(j) , ...
                '=temp(', num2str(1), ':', num2str(foldSizeCumSum(1)), ',:);']);
        else
            eval(['nfold.class' , num2str(a), '.fold', num2str(j) , ...
                '=temp(', num2str(foldSizeCumSum(j-1)+1), ':' , num2str(foldSizeCumSum(j)), ',:);']);
        end
    end
end
%For each unique group:
%Take the group as a hold out or test data set
%Take the remaining groups as a training data set
for a = 1:N   
    testData = []; testLabel = []; trainData = []; trainDataLabel = [];
    %test data
    for j = 1:size(class,1)
        eval(['testData = [testData; nfold.class' , num2str(j), '.fold', num2str(a), '];']);
        eval(['testLabel = [testLabel; j * ones(size(nfold.class', num2str(j), '.fold', num2str(a), ',1),1)];']);
    end
    for d=1:length(testLabel)
        if testLabel(d)<2
            testLabel(d)=0;%For healthy ppl
        else
            testLabel(d)=1;%for patients
        end
    end

    %train data                                      
    for j = 1:size(class,1)
        for h = 1:N
            if h ~= a
               eval(['trainData = [trainData; nfold.class' , num2str(j), '.fold', num2str(h), '];']);
               eval(['trainDataLabel = [trainDataLabel; j * ones(size(nfold.class', num2str(j), '.fold', num2str(h), ',1),1)];']);
            end
        end
    end
    for b=1:length(trainDataLabel)
        if trainDataLabel(b)<2
            trainDataLabel(b)=0;%For healthy ppl
        else
            trainDataLabel(b)=1;%for patients
        end
    end
       predData= myDisc(trainData,trainDataLabel,testData);
      for c=1:length(predData)
        if predData(c)<2
            predData(c)=0;%For healthy ppl
        else
            predData(c)=1;%for patients
        end
    end
disp("For fold "+a);
stats=confusionmatStats(predData,testLabel);

disp("--------------------------------------------------------------------------")

%eval(['confusionMatrix.fold', num2str(a), ' = confusionmat(testLabel, predData);']);
%if a == 1
%confusionMatrix.allFolds = confusionMatrix.fold1;
%else
%eval(['confusionMatrix.allFolds = confusionMatrix.allFolds + confusionMatrix.fold', num2str(a),';']);
end





%disp("For all folds-");
%disp("Confusion Matrix:");
%disp(confusionMatrix.allFolds);
%C=confusionMatrix.allFolds;
%confusionchart(confusionMatrix.allFolds);
%TP=C(1,1); 
%TN=C(2,2); 
%acc=(TP+TN)/totalSamples;
%disp("Mean Accuracy:"+acc);


function Lpred= myDisc(trainData,trainDataLabel,testData)
cl = unique(trainDataLabel);
giX=[];
for c = 1:length(cl)
PWi= length(trainDataLabel(trainDataLabel==cl(c))) / length(trainDataLabel);
UniqueCl= find(trainDataLabel == cl(c));
CovarianceMatrix= cov(trainData(UniqueCl,:));
mu= mean(trainData(UniqueCl,:))';

constW= -0.5*mu'*inv(CovarianceMatrix)*mu - log(det(CovarianceMatrix))/2 + log(PWi);
giX(:,c)  = diag(bsxfun(@plus, testData * -((inv(CovarianceMatrix))/2) * testData', ((inv(CovarianceMatrix)) * mu)' * testData' + constW));
disp(length(giX));
end
[~,Lpred] = max(giX,[],2);
end


