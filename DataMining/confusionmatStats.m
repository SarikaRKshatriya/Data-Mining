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
field1 = 'confusionMatrix';
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
disp(value2);
field3 = 'sensitivity';  value3 = sensitivity;
disp(field3);
disp(value3);
field4 = 'specificity';  value4 = specificity;
disp(field4);
disp(value4);
 
stats = struct(field1,value1,field2,value2,field3,value3,field4,value4);
 
end

