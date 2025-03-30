function [X, wm, wc] = generate_sigma_points(x, P, params)
%GENERATE_SIGMA_POINTS 生成UKF的sigma点
%   x: 状态向量
%   P: 状态协方差矩阵
%   params: UKF参数
%   X: 生成的sigma点
%   wm: 均值权重
%   wc: 协方差权重

% 提取UKF参数
n = params.n;         % 状态维度
alpha = params.alpha; % 缩放参数
beta = params.beta;   % 二次非线性参数
kappa = params.kappa; % 缩放参数

% 计算lambda
lambda = alpha^2 * (n + kappa) - n;

% 计算系数c
c = sqrt(n + lambda);

% 计算权重
wm = zeros(2*n + 1, 1);  % 均值权重
wc = zeros(2*n + 1, 1);  % 协方差权重

wm(1) = lambda / (n + lambda);
wc(1) = wm(1) + (1 - alpha^2 + beta);

for i = 2:2*n+1
    wm(i) = 1 / (2 * (n + lambda));
    wc(i) = wm(i);
end

% 计算矩阵平方根（使用Cholesky分解）
try
    % 尝试Cholesky分解
    L = chol((n + lambda) * P, 'lower');
catch
    % 如果失败，确保P是正定的
    fprintf('警告: 协方差矩阵不是正定的，添加对角增量\n');
    [V, D] = eig(P);
    D = diag(max(diag(D), 1e-6));  % 确保所有特征值为正
    P = V * D * V';
    P = (P + P') / 2;  % 确保对称性
    L = chol((n + lambda) * P, 'lower');
end

% 生成sigma点
X = zeros(n, 2*n+1);
X(:,1) = x;  % 中心点

for i = 1:n
    X(:,i+1) = x + L(:,i);
    X(:,i+1+n) = x - L(:,i);
end

% 返回sigma点和权重
end 