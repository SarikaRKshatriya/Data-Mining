clear all; close all; clc
N=5;
load Iris.csv;
class= [1,2,3];
Y =[class]; 
X = Iris (:,1:4);
Y = Iris (:,5); 
label = Y;
feat=X;

for k = 3:2:7
    for r = 1:2:5
        [accuracy, confusionMatrix] = cm(N, k, r);	
    end
end

function PredData = knn(trainData, trainDataLabel, testData, k, r)
class = unique(trainDataLabel);
PredData = [];
 
for iC = 1:length(class)
    cl = class(iC);
    idx = find(trainDataLabel==cl);
    eval(['dclass', num2str(iC), '=trainData(idx,:);']);
end
for i = 1:size(testData,1)
         t = testData(i,:);
         myKnnDist = [];
         for iC = 1:length(class)
             cl = class(iC);
             A = eval(['dclass', num2str(cl)]);
             s = size(A,1); 
             B=repmat(t,s,1);
			%call minkowski distance functio

             mdist= minkowskiDistance(A,B,r);
             sortData = sort(mdist);                   
             MyKnnDist = mode(sortData(1:k));                                                                                %find most repeated 
             myKnnDist = [myKnnDist;MyKnnDist,cl];
         end
         [vmin,imin] = min(myKnnDist(:,1));
         p = myKnnDist(imin,2);
         PredData = [PredData; p]; 
    end
 
end


function [accuracy, confusionMatrix] = cm(N, k, r)
N=5;
load Iris.csv;
class= [1,2,3];
Y =[class]; 
X = Iris (:,1:4);
Y = Iris (:,5);
label = Y;

feat=X;

accuracyMatrix = []; confusionMatrix = [];
 
data_nfold = divide_nfold_data(feat, label, N); 
C = unique(label);
for ifold = 1:N 
       %----prepare cross-validation training and testing dataset---% 
       idx_test = ifold; % index for testing fold
       idx_train = setdiff(1:N, ifold); % index for training folds
       Dtest = []; Ltest = []; % initialize testing data and label
       Dtrain = []; Ltrain = []; % initialize testing data and label
       
       %---construct the training and testing dataset for the ith fold cross validatoin
       for iC = 1:length(C) 
           cl = C(iC);   
           dtest = eval(['data_nfold.class',num2str(iC), '.fold', num2str(ifold)]);
           Dtest = [Dtest; dtest]; 
           Ltest = [Ltest; cl*ones(size(dtest,1), 1)]; 

           for itr = 1:length(idx_train)
               idx = idx_train(itr); 
               dtrain = eval(['data_nfold.class',num2str(iC), '.fold', num2str(idx)]);
               Dtrain = [Dtrain; dtrain];
               Ltrain = [Ltrain; cl*ones(size(dtrain,1), 1)]; 
           end 
          
       end
           Lpred=[];
                Lpred = knn(Dtest,Ltrain,Dtrain, k, r); 

 
 											   % KNN
    %testPredLabel = knn(testFeat,[trainFeat trainLabel],k,r);
    %knn(trainFeat, trainLabel, testFeat, k, p);
    
   									 		%Accuracy
    accuracy = sum(Lpred == Ltest)/length(Ltest);
    accuracyMatrix = [accuracyMatrix; accuracy];
end
accuracy = mean(accuracyMatrix);
%Confusion-matrix
confusionMatrix = confusionmat (Ltest,Lpred);
disp(accuracy);
disp(confusionMatrix);
end
function m = minkowskiDist(X,Y,r)

m=(sum((abs(X(i,:)-Y)).^r).^(1/r));

end
function data_nfold = divide_nfold_data(feature, label, N)
% This is to split a dataset into N-fold for cross-valiation purpose
% feature: the data matrix, each row is a smaple, each column is an attribute
% label: class label of the samples
% N: divide dataset into N parts with equal size

C = unique(label); 
for iC = 1:length(C)
    cl = C(iC); 
    idx = find(label==cl); 
    data = feature(idx,:);
    L = length(idx);     
    feat_nfold = nfold_set(data, N); 
    eval(['data_nfold.class', num2str(cl), '=feat_nfold;']);     
end
end


%% The function to find points to seperate dataset to N-fold 
function feat_nfold = nfold_set(feat, N)
% Determin the size of each subset
L = size(feat,1); % number of samples
n = floor(L/N);   % basic subset size 
rem = mod(L, N);  % Modulus after division, there are some extra samples to assign
a = n*ones(N,1);
if rem>0
  b = nchoosek(1:N,rem);
  c = ceil(rand*size(b,1));
  idx = b(c,:); 
  a(idx)= a(idx) + 1; 
end 
nfoldpt =[0; cumsum(a)];
nint = [nfoldpt(1:end-1)+1, nfoldpt(2:end)];  

for i = 1:N
    dsub = feat(nint(i,1):nint(i,2), :); 
    eval(['feat_nfold.fold', num2str(i), '=dsub;']); 
end
end
