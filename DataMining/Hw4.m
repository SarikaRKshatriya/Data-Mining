clear all; close all; clc
load BreastCancer.csv;
class= [1,2];
Y =[class]; 
X = BreastCancer (:,1:9);
Y = BreastCancer (:,10);
totalSamples=116;
%-----------------------------------------------------------------------------------
%assign binary values 1 and -1
for i=1:116
    if Y(i)<2
        Y(i)=1;%For healthy ppl
    else
        Y(i)=-1;%for patients
    end
end

N=5;
accuracyMatrix = []; confusionMatrix = [];
% 5 fold 
%Divide the data into 5 sets in each class
class = unique(Y);
for i = 1:size(class,1)
    NoOfFolds = [];
    temp = X(Y==class(i),:);
    index = find(Y==class(i));
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
            eval(['nfold.class' , num2str(i) , '.fold', num2str(j) , ...
                '=temp(', num2str(1), ':', num2str(foldSizeCumSum(1)), ',:);']);
        else
            eval(['nfold.class' , num2str(i), '.fold', num2str(j) , ...
                '=temp(', num2str(foldSizeCumSum(j-1)+1), ':' , num2str(foldSizeCumSum(j)), ',:);']);
        end
    end
end
%For each unique group:
%Take the group as a hold out or test data set
%Take the remaining groups as a training data set
for i = 1:N   
    testData = []; testLabel = []; trainData = []; trainDataLabel = [];
    %test data
    for j = 1:size(class,1)
        eval(['testData = [testData; nfold.class' , num2str(j), '.fold', num2str(i), '];']);
        eval(['testLabel = [testLabel; j * ones(size(nfold.class', num2str(j), '.fold', num2str(i), ',1),1)];']);
    end
    %train data                                      
    for j = 1:size(class,1)
        for h = 1:N
            if h ~= i
               eval(['trainData = [trainData; nfold.class' , num2str(j), '.fold', num2str(h), '];']);
               eval(['trainDataLabel = [trainDataLabel; j * ones(size(nfold.class', num2str(j), '.fold', num2str(h), ',1),1)];']);
            end
        end
    end
    %Using Derived Bayesian Decision Rule
    % if Ratio of likelihood > [(lambda12-lambda22)/(lambda21-lambda11)]*ratio of prior
    predData= myBayDecBndry(trainData,trainDataLabel,testData);
    % predData= myBayDecBndry(trainData,trainDataLabel,testData);
     %predData= myFisherDisc(trainData,trainDataLabel,testData);

disp("For fold "+i);

stats=confusionmatStats(predData,testLabel);

disp("--------------------------------------------------------------------------")

eval(['confusionMatrix.fold', num2str(i), ' = confusionmat(testLabel, predData);']);
if i == 1
confusionMatrix.allFolds = confusionMatrix.fold1;
else
eval(['confusionMatrix.allFolds = confusionMatrix.allFolds + confusionMatrix.fold', num2str(i),';']);
end



end

disp("For all folds-");
disp("Confusion Matrix:");
disp(confusionMatrix.allFolds);
C=confusionMatrix.allFolds;
confusionchart(confusionMatrix.allFolds);
TP=C(1,1); 
TN=C(2,2); 
acc=(TP+TN)/totalSamples;
disp("Mean Accuracy:"+acc);
%area(C);
function pre= myBayDecBndry(trainData,trainDataLabel,testData)
   C = unique(trainDataLabel); 
   pre = []; 
   lambda11=0;
   lambda22=0;
   lambda12=2;
   
   lambda21=1;
   
   rhslambda=(lambda12-lambda22)/(lambda21-lambda11);
   for iC = 1:length(C) % For each class i, calculate P(X|Wj)P(Wj) for all testing samples
       cl = C(iC);  
       idx = find(trainDataLabel==cl); 
       data = trainData(idx,:); 
       mu = mean(data); % feature mean vector
       sigma = cov(data); % feature covariance matrix
       P = length(idx)/length(trainDataLabel);
      
       % For each testing sample, calculate P(X|Wj)P(Wj) = likelihood of class i * prior of class i
       for j = 1:size(testData,1) 
           x = testData(j, :); 
           likelihood = mvnpdf(x,mu,sigma); % likelihood of the current class i
           likelihood
           prior = P; % prior of the current class i
           %likelihood(j) 
           % Record values of the discriminat function G(X)
           % In the following matrix G, each row represent a class, and
           % each column represent a testing sample
          % pxw(iC) = likelihood(j); % P(X|Wj)P(Wj) 
       end
    
        l(iC)=max(likelihood);
        pw(iC)=max(prior);
        
   %l(iC)
   
  rhsPw=pw(1)/pw(2);
  rhsPw
   lhs=l(1)/l(2);
   rhs=rhslambda*rhsPw;
   
   if  lhs > rhs
        pre=[pre;1];
   else
       pre=[pre;2];
   end
  % lhs
   % For each testing sample, find the index of the class that have maximum
   % value of likelihood*prior
  % [~, pred] = max(pxw);  
   end
  pre = C(pre);    
end 


function pre= myFisherDisc(trainData,trainDataLabel,testData)
l=length(testData);
p=zeros(1,l);

    r=1;
    cl = unique(trainDataLabel);
    NumberOfClasses = 2;
     localmu1=0;
localmu2=0;
CovarianceMatrix1=0;
CovarianceMatrix2=0;
projectedSample=[];
for c = 1:length(cl)
UniqueCl= find(trainDataLabel == cl(c));
CovarianceMatrix= cov(trainData(UniqueCl,:));
mu= mean(trainData(UniqueCl,:))';
if c==1
    localmu1=mu;
    CovarianceMatrix1=CovarianceMatrix;
end
if c==2
    localmu2=mu;
    CovarianceMatrix2=CovarianceMatrix;
end   
CovarianceMatrix=CovarianceMatrix1+CovarianceMatrix2;
ps=inv(CovarianceMatrix)+(localmu1-localmu2);
projectedSample = [projectedSample;ps]
end
ps=max(projectedSample,[],2);
%ps(numel(p)) = 0;
%ps = [ps, zeros(1, length(p) - length(ps))];

pre=[];
for i=1:length(testData)
     
    if ps(i)>0
        pre = [pre;1]
    end
    if ps(i)<=0
       pre = [pre;2] 
    end
   
end

end
function predData = middleLine(trainData,trainDataLabel,testData)
   C = unique(trainDataLabel); 
   
    predData=[];
   for iC = 1:length(C) 
       cl = C(iC);
       idx = find(trainDataLabel==cl); 
       data = trainData(idx,:); 
       
       mu = mean(data)'; 
       E = cov(data);
       P = length(idx)/length(trainDataLabel); 
       W = -0.5*inv(E); 
       w = inv(E)*mu; 
       w0 = -0.5*mu'*inv(E)*mu-0.5*log(det(E))+log(P); 
       
       for j = 1:size(testData,1)
           x = testData(j, :)'; 
           % G(X) 
           G(iC, j) =x'*W*x + w'*x + w0; 
             
       end
   end
  
    [~, predData] = max(G);
  
    predData = C(predData); 


end


function stats = confusionmatStats(group,grouphat)
field1 = 'confusion_Matrix';
if nargin < 2
    value1 = group;
else
    [value1,gorder] = confusionmat(group,grouphat);
end
disp(field1);
C=confusionmat(group,grouphat);
disp(C);
numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));
[TP,TN,FP,FN,accuracy,sensitivity,specificity] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   TP(class) = value1(class,class);
   tempMat = value1;
   tempMat(:,class) = []; % remove column
   tempMat(class,:) = []; % remove row
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(value1(:,class))-TP(class);
   FN(class) = sum(value1(class,:))-TP(class);
end
for class = 1:numOfClasses
    accuracy(class) = (TP(class) + TN(class)) / totalSamples;
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    specificity(class) = TN(class) / (FP(class) + TN(class));

end
disp("For each class:");
field2 = 'accuracy'; 
value2 = accuracy;
disp(field2);
disp(value2);
disp("Mean Accuracy: ");
disp(mean(value2));
stats = struct(field1,value1,field2,value2);

end