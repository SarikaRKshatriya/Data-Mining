%homework 4 :problem 2
%Perform ROC analyses for the three classification models developed in problem 1 
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
        Y(i)=-1;%For healthy ppl
    else
        Y(i)=1;%for patients
    end
end
%used the first 38 samples in Class 1 and the first 48 samples in Class 2 to construct training dataset, 
%and the remaining samples for testing dataset. 

trainDataA=X (1:38,:);
trainLabelA=Y (1:38,:);
testDataA=X (39:52,:);
testLabelA=Y (39:52,:);

trainDataB=X (53:101,:);
trainLabelB=Y (53:101,:);
testDataB=X (102:116,:);
testLabelB=Y (102:116,:);

trainData=[trainDataA ;trainDataB];
testData=[testDataA ;testDataB];
trainLabel=[trainLabelA; trainLabelB];
testLabel=[testLabelA ;testLabelB];

lambda = [0 1; 1 0]; 
option = 2; 
   %threshold_list = -75:1:75; 
   
   % Lpred = FishersLDA(Dtrain, Ltrain, Dtest, lambda, option);
   % The updated function FishersLDA_v2 can calculated AUC value
   [Lpred, w, AUC, ROC, senspe] =  FishersLDA_v2(Dtrain, Ltrain, Dtest, Ltest, lambda, option);
   %---------------------------------------------------------%

   %---Calculate Classification Accuracy-----%
   acc = sum(Lpred==Ltest)/length(Ltest);  
   
   %---Calculate Sensitivity & Specificity based on Lpred and Ltest-----%
   idx1 = find(Ltest==1); pred1 = Lpred(idx1); 
   sen = length(find(pred1==1))/length(idx1); 
   idx2 = find(Ltest==-1); pred2 = Lpred(idx2); 
   spe = length(find(pred2==-1))/length(idx2);



