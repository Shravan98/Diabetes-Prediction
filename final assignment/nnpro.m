data = csvread('diabetes.csv', 1);
%features will contain all the features
siz = size(data);
features = data(:, 1:siz(2)-1);
%output is 1 or 0 depending on if the patient has diabetes
output = data(:,siz(2));
%calculating eigen value and eigenvectors of the covariance matrix
%the eigenvalue here are in increasing order and eigenvectors are arranged
%correspondingly.magnitude of eigenvalue=variance
[coeff, score, latent] = pca(features);
%extract the number of top featues from the pca having maximum variance
%number of features removed
rem = 2;
features = score(:, 1:siz(2)-rem-1);
%training the neural network using different models
training('trainlm', features, output)
training('trainbfg', features, output)
training('trainrp', features, output)