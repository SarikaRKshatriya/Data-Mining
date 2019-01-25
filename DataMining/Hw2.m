%Classification Using decision tree
clear all;
close all;
clc
N=5;
[accuracy, confusionMatrix] = cm(N);	

% Calculate confusion matrix and accuracy for N fold
function [accuracy, confusionMatrix] = cm(N)
%N=5;
load Iris.csv;
class= [1,2,3];
 
X = Iris (51:end,3:4);
Y = Iris (51:end,5);
accuracyMatrix = []; confusionMatrix = [];
% 5 fold 
%Divide the data into 5 sets in each class
%class = unique(Y);
for i = 2:3
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
    for j = 2:3
        eval(['testData = [testData; nfold.class' , num2str(j), '.fold', num2str(i), '];']);
        eval(['testLabel = [testLabel; j * ones(size(nfold.class', num2str(j), '.fold', num2str(i), ',1),1)];']);
    end
    %train data   									 
    for j = 2:3
        for h = 1:N
            if h ~= i
               eval(['trainData = [trainData; nfold.class' , num2str(j), '.fold', num2str(h), '];']);
               eval(['trainDataLabel = [trainDataLabel; j * ones(size(nfold.class', num2str(j), '.fold', num2str(h), ',1),1)];']);
            end
        end
    end
   % disp(trainData);
   %disp(testData);
   % len=length(trainData);
   % disp(len);
    % prdict using KNN function
     %for k=1:20
        PredData=myFunction(trainData,trainDataLabel,testData);
       % disp(testData(k:k,1)+","+ testData(k:k,2));
     %end
    disp(PredData);
    disp(length(PredData));
 	%Accuracy
    accuracy = sum(PredData == testLabel)/length(testLabel);
   % accuracyMatrix for all folds
   AccuracyMatrix = [];
eval(['confusionMatrix.fold', num2str(i), ' = confusionmat(testLabel, PredData);']);
if i == 1
confusionMatrix.allFolds = confusionMatrix.fold1;
else
eval(['confusionMatrix.allFolds = confusionMatrix.allFolds + confusionMatrix.fold', num2str(i),';']);
end
AccuracyMatrix = [AccuracyMatrix; accuracy];

disp(confusionMatrix.allFolds);
figure
confusionchart(confusionMatrix.allFolds);
end
figure
gscatter(testData(:,1),testData(:,2), PredData,'rg');
hold on;
xrange = [15 60];
yrange = [5 25];
%Assumed the boundary on Petal Length is 4.8, and 
%the decision boundary on Petal Width is 1.7. 
%hline = refline([0 1.7]);
%hline.Color = 'r';
%hold on
%line([4.8 4.8], [0 1.7])
%title('Red:Versicolor Green:Virginica');
%xlabel('Petal Length')
%ylabel('Petal Width')
%Calculate Confusion-matrix 
confusionMatrix = confusionmat (testLabel,PredData);
figure
cf='Confusion Matrix';
%disp(cf);
disp(confusionMatrix);
%confusionchart(confusionMatrix);
%stats=confusionmatStats(PredData,testLabel);

end
%-------------------------------------------------------------------------%
function flowerType =myFunction(trainData,trainDataLabel,testData)
flowerType =[];
len=length(trainData);
for k=1:len
    trainPL=trainData(k:k,1);
    trainPW=trainData(k:k,2);
    if trainPW < 1.7
        if trainPL < 4.8
            if trainPW < 1.65
                getFlowerType2=trainDataLabel(k);
            elseif  trainPW >= 1.65
                getFlowerType3 = trainDataLabel(k)
            end
            elseif trainPL >= 4.8 && trainPL <=4.9
                    getFlowerType2=trainDataLabel(k);
        else
                    getFlowerType3=trainDataLabel(k);
        end
    else
        getFlowerType3=trainDataLabel(k); 
    end
end
testlen=length(testData);

for t=1:testlen
 pl=testData(t:t,1);
 pw=testData(t:t,2);
    if pw < 1.7 %the decision boundary on Petal Width is 18
        if pl< 4.8 %the binary decision boundary on Petal Length is 4.3
            if pw< 1.65
                f=getFlowerType2;
        elseif pw>=1.65
                f=getFlowerType3;
            end
        elseif pl>=4.8 && pl<=4.9
            f=getFlowerType2;
        else
            f=getFlowerType3;
        end
    else
        f=getFlowerType3;
    end
    flowerType = [flowerType; f]; 
end
 

end





