clear all; clc; close all; 

clear all;
close all;
clc
load Iris.csv;
X = Iris (51:end,3:4);
Y = Iris (51:end,5); 
PL=Iris (51:end,3);
PW=Iris (51:end,4);
accuracyMatrix = []; 


N = 5; % N-fold cross validation 
class = unique(Y); %extract label information from label vector

for i = 2:size(class,1)
    foldSize = [];
    temp = X(Y==class(i),:);
    classIndex = find(Y==class(i));
    classCount = numel(classIndex);
    %Shuffle the dataset randomly.
    rng('shuffle');
    temp = temp(randperm(classCount),:);
    quotient = floor(classCount/N);
    remainder = mod(classCount, N);
    rng('shuffle');
    extra = randsample(1:N,remainder);
    foldSize = ones(N,1) * quotient;
    foldSize(extra) = foldSize(extra) + 1;
    %Cumulative Sum
    foldSizeCum = cumsum(foldSize);
    %Split the dataset into 5 groups
    for j = 1:N
        if j==1
            eval(['nfold.class' , num2str(i) , '.fold', num2str(j) , ...
                '=temp(', num2str(1), ':', num2str(foldSizeCum(1)), ',:);']);
        else
            eval(['nfold.class' , num2str(i), '.fold', num2str(j) , ...
                '=temp(', num2str(foldSizeCum(j-1)+1), ':' , num2str(foldSizeCum(j)), ',:);']);
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

         disp(testLabel);  
       %---------------------------------------------------------%
       %Decision tree function
        for k=1:10
        Lpred(k)=myFunction(testData(k:k,1),testData(k:k,2));
        end
      %---Calculate Classification Accuracy-----%
       acc = sum(Lpred==testLabel)/length(testLabel);  
       %disp(acc);
      
end  
confusionMatrix = confusionmat (Lpred,testLabel);
%disp(confusionMatrix);
%confusionchart(confusionMatrix);
%scatter graph for test data 
gscatter(testData(:,1),testData(:,2), testLabel,'rg');
hold on;
xrange = [15 60];
yrange = [5 25];
%Assumed the boundary on Petal Length is 4.8, and 
%the decision boundary on Petal Width is 1.7. 
hline = refline([0 1.7]);
hline.Color = 'r';
hold on
line([4.8 4.8], [0 1.7])
title('Red:Versicolor Green:Virginica');
xlabel('Petal Length')
ylabel('Petal Width')


%function which takes input as Lpred,Ltest and gives output with
%confusion matrix,accuracy,sensitivity and specificity.
%used this function from mathworks fileexcahnge 
stats=confusionmatStats(Lpred,testLabel);




function flowerType=myFunction(pl,pw)
if pw < 1.7 %the decision boundary on Petal Width is 18
    if pl< 4.8 %the binary decision boundary on Petal Length is 4.3
        if pw< 1.65
            flowerType=2;
        elseif pw>=1.65
            flowerType=3;
        end
    elseif pl>=4.8 && pl<=4.9
        flowerType=2;
    else
        flowerType=3;
    end
else
    flowerType=3;
end
end

function stats = confusionmatStats(group,grouphat)
% INPUT
% group = true class labels
% grouphat = predicted class labels
%
% OR INPUT
% stats = confusionmatStats(group);
% group = confusion matrix from matlab function (confusionmat)
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____| 
%       Actual  p |     |     |
%      Classes  n |     |     |
%
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 
field1 = 'confusionMat';
if nargin < 2
    value1 = group;
else
    [value1,gorder] = confusionmat(group,grouphat);
end
disp(field1);
disp(confusionmat(group,grouphat));
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
  %  precision(class) = TP(class) / (TP(class) + FP(class));
 %   f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
end
field2 = 'accuracy';  value2 = accuracy;
disp(field2);
disp(value2(1));
field3 = 'sensitivity';  value3 = sensitivity;
disp(field3);
disp(value3(1));
field4 = 'specificity';  value4 = specificity;
disp(field4);
disp(value4(1));

stats = struct(field1,value1,field2,value2,field3,value3,field4,value4);

end

