function [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params)
%COMPUTE_STEERING_VECTOR 计算给定方位角和俯仰角的收发导向矢量
%   tx_array: 发射阵列结构体
%   rx_array: 接收阵列结构体
%   r: 距离 (m)
%   az: 方位角 (度)
%   el: 俯仰角 (度)
%   params: 系统参数结构体
%   a_tx: 发射阵列导向矢量
%   a_rx: 接收阵列导向矢量

% 提取波长
lambda = params.c / params.fc;  % 波长

% 计算单位方向向量
az_rad = deg2rad(az);
el_rad = deg2rad(el);
k_vec = 2*pi/lambda * [cosd(el)*cosd(az); cosd(el)*sind(az); sind(el)];

% 计算发射阵列导向矢量
a_tx = compute_array_steering_vector(tx_array.elements_pos, k_vec);

% 计算接收阵列导向矢量
a_rx = compute_array_steering_vector(rx_array.elements_pos, k_vec);

end

function a = compute_array_steering_vector(elements_pos, k_vec)
    % 计算阵列导向矢量
    % elements_pos: Nx3矩阵，每行表示一个阵元的[x,y,z]坐标
    % k_vec: 波数向量
    
    % 计算每个阵元的空间相位
    phase = zeros(size(elements_pos, 1), 1);
    for i = 1:size(elements_pos, 1)
        % 计算k·r (点积)
        phase(i) = k_vec(1) * elements_pos(i, 1) + ...
                   k_vec(2) * elements_pos(i, 2) + ...
                   k_vec(3) * elements_pos(i, 3);
    end
    a = exp(1j * phase);
    
    % 归一化导向矢量
    a = a / norm(a);
end 