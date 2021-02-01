clear ; close all; clc
load ('ex7faces.mat');

figure;
for i=1:16
subplot(4,4,i)
ex_image = X(i*101 + i^2, :);
image = reshape(ex_image, 32, 32);
colormap(gray);
imagesc(image);
end

K_set = [512 256 128 64 32 16 8 4 2 1];

% How much variation each PC captures from the data.
var_perc = []; 


for k = 1:length(K_set)

[m, n] = size(X);  % m # of examples, n # of features (variables)

% Zero mean and normalization
mu = mean(X);
X_zeromean = X - mu;
sigma = std(X_zeromean);
X_norm = X_zeromean./sigma;

% SVD
Cov_mtx = (1/m).*X_norm'*X_norm;  
[U,S,V] = svd(Cov_mtx); % Principal components 

% Project data
K = round(1024/K_set(k));
U_reduced = U(:,1:K);
Z = X_norm * U_reduced;

% Recover the approximation data back with error. Which means we are
% visualizing data on the line which spanned by principal components
U_reduced = U(:,1:K);
X_rec = Z*U_reduced';

% Revert the preprocessing step
X_rec = X_rec.*sigma;
X_rec = X_rec + mu;

figure;
for i=1:16
subplot(4,4,i)
ex_image = X_rec(i*101 + i^2, :);
image = reshape(ex_image, 32, 32);
colormap(gray);
imagesc(image);
end

tit = sprintf('The K value is %d', K);
sgtitle(tit);




end






