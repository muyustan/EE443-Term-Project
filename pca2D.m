clear ; close all; clc

load ('ex7data1.mat');

X(:, 1) = X(:, 1) * 15;
X(:, 2) = X(:, 2) * 30;

figure;
plot(X(:, 1), X(:, 2), 'ok');
axis([-30 150 60 240]); axis square;
xlabel('weight');
ylabel('height');
% axis([0.5 6.5 2 8]); axis square;

[m, n] = size(X);  % m # of examples, n # of features (variables)

% Zero mean
mu = mean(X);
X_zeromean = X - mu;

% SVD
Cov_mtx = (1/m) .* X_zeromean' * X_zeromean;  
[U,S,V] = svd(Cov_mtx); % Principal components 

%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.
hold on;
drawLine(mu, mu +  .05*S(1,1) * U(:,1)', '-b', 'LineWidth', 2);
drawLine(mu, mu +  .05*S(2,2) * U(:,2)', '-r', 'LineWidth', 2);
legend('Data points', 'Principle Component 1', 'Principle Component 2');
hold off;

% How much of the variance is captured by these principal components?
% S has the variance associated with that eigenvector in its diagonal
% entries.

total_var = trace(S); % get the total variance associated with the data set
percent_var = 100 * [S(1, 1) ./ total_var, S(2, 2) ./ total_var];
figure;
bar(percent_var);
xlabel('Principal Component');
ylabel('Variation (%) along');


% draw the zero mean version of the original data first
figure;
plot(X_zeromean(:, 1), X_zeromean(:, 2), 'bo');
axis([-80 80 -80 80]); axis square;


% Project data
K = 1;
U_reduced = U(:,1:K);
Z = X_zeromean * U_reduced;

% Recover the approximation data back.
% Which means we are projecting the data points on the line
% which is essentially the first principal component.
U_reduced = U(:,1:K);
X_rec = Z*U_reduced';

hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_zeromean, 1)
    drawLine(X_zeromean(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end

legend('zero mean original data', 'projection onto PC1');

hold off;

% plot the 1D reduced points on PC1

figure;
scatter(Z, zeros(1, length(Z)), 'ro');
xlabel('PC1');



function drawLine(p1, p2, varargin)
%DRAWLINE Draws a line from point p1 to point p2
%   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
%   current figure

plot([p1(1) p2(1)], [p1(2) p2(2)], varargin{:});

end
