
load Iris.csv;

Class= [1,2,3];
%Y =[Class];
X = Iris (:,1:4);
Y = Iris (:,5); 
Sepal_length = Iris(:,1);
Sepal_width = Iris (:,2);
Petal_length = Iris (:,3);
Petal_width = Iris (:,4);

class = Y;
names = char('Sepal length','Sepal width','Petal length','Petal width');
%2d scatter plot for the four attributes
gplotmatrix(X,[],Y,'','',[],'on','none',names,[]);

%3d scatter plot
scatter3(Sepal_length(1:50),Sepal_width(1:50),Petal_width(1:50),'r');
hold on;
scatter3(Sepal_length(50:100),Sepal_width(50:100),Petal_width(50:100),'g');
hold on;
scatter3(Sepal_length(100:150),Sepal_width(100:150),Petal_width(100:150),'magenta');

title('3D scatter plot of three attributes ');

xlabel('Sepal Length');
ylabel('Sepal Width');
zlabel('Petal Width');


V=[sepal_length,sepal_width,petal_length,petal_width];
imagesc(V);
colorbar;

%Histogram for Sepal_length for the 3 classes
h1= Iris(1:50,1);
h2= Iris(51:100,1);
h3= Iris(101:150,1);
histogram(h1);
hold on;
histogram(h2);
hold on;
histogram(h3);
legend('setosa','versicolor','verginica')

%Histogram for Sepal_width for the 3 classes
h1= Iris(1:50,2);
h2= Iris(51:100,2);
h3= Iris(101:150,2);
histogram(h1);
hold on;
histogram(h2);
hold on;
histogram(h3);
legend('setosa','versicolor','verginica')

%Histogram for Petal_length for the 3 classes
h1= Iris(1:50,3);
h2= Iris(51:100,3);
h3= Iris(101:150,3);
histogram(h1);
hold on;
histogram(h2);
hold on;
histogram(h3);
legend('setosa','versicolor','verginica')

%Histogram for Petal_width for the 3 classes
h1= Iris(1:50,4);
h2= Iris(51:100,4);
h3= Iris(101:150,4);
histogram(h1);
hold on;
histogram(h2);
hold on;
histogram(h3);
legend('setosa','versicolor','verginica')

% Boxplot for the Sepal_length
 boxplot(sepal_length,Y);

 % Boxplot for the Sepal_width
 boxplot(Sepal_width,Y);

 % Boxplot for the Petal_length
 boxplot(Petal_length,Y);

  boxplot(Petal_width,Y);
  
  imagesc(corrcoef(X));
  
  %Parallel coordinates plot of the four attributes
parallelcoords(Iris(:,1:4),'group',Y,'labels',names);

function [MD] = minkowskiDistance(A,B,r)
MD=(sum((abs(A-B)).^r).^(1/r));
end


function [tstatd] = tStasticsDistance(A,B)
tstatd = ((abs(mean(A)-mean(B)))/std(A-B));
end


function [ MD ] = mahalanobis(A,B,M)
P = inv(M);
MD = (A-B)*P*(transpose(A-B));
end

%Minkowski Distance with r=1
load Iris.csv;
A = Iris (:,1:4);
B = [5.0000,3.5000,1.4600,0.2540];
r=1;
MD=minkowskiDist(A,B,r);
plot(MD);
function [MD] = minkowskiDist(A,B,r)
for i=1:150
MD(i)=(sum((abs(A(i,:)-B)).^r).^(1/r));
end
end

%Plot the generated two time series in one plot
filename = 'HW1_DataMining.txt ';
A= importdata(filename);
plot(A);
title('Two time series in one plot');
legend('Series 1','Series 2');

% Converted the given text file to csv file
load HW1_DataMining.csv;
Series1 = HW1_DataMining(:,1);
Series2 = HW1_DataMining (:,2);
correlation=corr(S1,S2);
disp(correlation);

%Normalize the feature matrix of the IRIS dataset 
load Iris.csv;
X = Iris (:,1:4);
%Z and N gives same results
%Z=zscore(X);
%disp(Z);
%or
N= normalize(X);
disp(N);
plot(N);













