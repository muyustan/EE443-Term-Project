clear ; close all; clc

NUM_DATA_POINTS = 200;

var1 = randi([-25 -10], NUM_DATA_POINTS/2, 1); % var1 values
var2 = randi([-15 -10], NUM_DATA_POINTS/2, 1); % var2 values 
var3 = randi([0 5], NUM_DATA_POINTS/2, 1); % var3 values 

X = [var1 var2 var3];

var1 = randi([-5 25], NUM_DATA_POINTS/2, 1); % var1 values
var2 = randi([0 15], NUM_DATA_POINTS/2, 1); % var2 values 
var3 = randi([-10 -5], NUM_DATA_POINTS/2, 1); % var3 values 

X = [X;
      var1 var2 var3];

X(1:NUM_DATA_POINTS/2, 3) = X(1:NUM_DATA_POINTS/2, 3) + 5; 

[m, n] = size(X);  % m # of examples, n # of features (variables)

figure;
scatter3(X(:,1), X(:,2), X(:,3));
xlabel('var1');
ylabel('var2');
zlabel('var3');
xlim([-40 40]);
ylim([-40 40]);
zlim([-40 40]);
hold on;

% pause;

% Zero mean
mu = mean(X);
X_zeromean = X - mu;
% sigma = std(X_zeromean);
% X_norm = X_zeromean./sigma;

% SVD
Cov_mtx = (1./m) .* X_zeromean' * X_zeromean;  
[U, S, V] = svd(Cov_mtx); % Principal components 

%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.

drawLine3D(mu', mu' +  .05*S(1,1) .* U(:,1), '-b', 'LineWidth', 2);
drawLine3D(mu', mu' +  .25*S(2,2) .* U(:,2), '-r', 'LineWidth', 2);
drawLine3D(mu', mu' +  .5*S(3,3) .* U(:,3), '-g', 'LineWidth', 2);
legend('Data points', 'Principle Component 1(1/20)',...
     'Principle Component 2(1/4)', ...
     'Principle Component 3(1/2)');
hold off;

% pause;

% Project the data on PC1 PC2 plane
K = 2;
% get only first 2 principle components
U_reduced = U(:, 1:K);

% y is the results in the new 2 dimension space(PC1 PC2 plane)
y = U_reduced' * (X_zeromean)';
y = y';

% Recover the approximation data back with error. Which means we are
% visualizing data on the plane which spanned by principal components
U_reduced = U(:, 1:K);
X_rec = y*U_reduced';
X_rec = X_rec + mu;

figure;
scatter3(X(:,1), X(:,2), X(:,3));
zlabel('var3');
ylabel('var2');
xlabel('var1');
zlim([-40 40]);
hold on;

% pause;

scatter3(X_rec(:, 1), X_rec(:, 2), X_rec(:, 3));


for i = 1:size(X, 1)
    drawLine3D(X(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end

legend('original points', 'projection onto PC1 PC2 plane');
hold off;

% pause;

% plot the data on the new coordinate system spanned by PC1 and PC2

figure;
scatter(y(:, 1), y(:,2), 'ko');
xlim([-40 40]);
ylim([-40 40]);
xlabel('PC1');
ylabel('PC2');


% FUNCTIONS %

function drawLine3D(p1, p2, varargin)
%DRAWLINE Draws a line from point p1 to point p2
%   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
%   current figure

plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], varargin{:});

end