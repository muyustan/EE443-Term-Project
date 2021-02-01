clear ; close all; clc
load ('ex7faces.mat');

figure;
for i=1:16
subplot(4,4,i)
ex_image = X(101*i + i^2, :);
image = reshape(ex_image, 32, 32);
colormap(gray);
imagesc(image);
end


% How much variation each PC captures from the data.
var_perc = []; 


[m, n] = size(X);  % m # of examples, n # of features (variables)

% Zero mean and normalization
mu = mean(X);
X_zeromean = X - mu;
sigma = std(X_zeromean);
X_norm = X_zeromean./sigma;

% SVD
Cov_mtx = (1/m).*X_norm'*X_norm;  
[U,S,V] = svd(Cov_mtx); % Principal components 

total_var = trace(S);

opt_k = 1;
thresh_vec = [];
THRESHOLD = 99;
for j = 1:n
    
    
    thresh = 100*trace(S(1:j, 1:j))./total_var;
    thresh_vec(end+1) = thresh;
    
    if thresh > THRESHOLD
        continue;
    else
        opt_k = opt_k +1;
    end
    
end

figure;
x_ax = 1:n;
plot(x_ax,thresh_vec, 'LineWidth', 2);
hold on;
plot(opt_k, thresh_vec(opt_k), 'X','LineWidth', 3);
plot(x_ax,ones(1, length(x_ax))*THRESHOLD, '--', 'LineWidth', 2);
ylim([0 110]);
grid on;
title('How much variation captured by first k Principle Component');
legend('Cumulative Variation','Optimum k', 'Threshold');
xlabel('k');
ylabel('Variation (%)');
hold off;


% Project data
K = opt_k;
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
ex_image = X_rec(101*i + i^2, :);
image = reshape(ex_image, 32, 32);
colormap(gray);
imagesc(image);
end


