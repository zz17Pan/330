function [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params)
%COMPUTE_STEERING_VECTOR 高精度导向矢量计算
%   实现亚度级角度精度的导向矢量计算

% 参数检查和预处理
if r < 0.1
    warning('距离过小，设置为0.1m');
    r = 0.1;
end

% 提取波长
lambda = params.c / params.fc;

% 高精度角度计算
az_rad = deg2rad(az);
el_rad = deg2rad(el);

% 使用数值稳定的三角函数计算
cos_el = cos(el_rad);
sin_el = sin(el_rad);
cos_az = cos(az_rad);
sin_az = sin(az_rad);

% 计算精确的方向向量
k_dir = [cos_el*cos_az; cos_el*sin_az; sin_el];

% 确保单位向量
k_norm = norm(k_dir);
if k_norm < eps
    warning('方向向量接近零，添加微小扰动');
    k_dir = k_dir + eps;
    k_norm = norm(k_dir);
end
k_dir = k_dir / k_norm;

% 计算波数向量
k_vec = 2*pi/lambda * k_dir;

% 高精度阵列导向矢量计算
a_tx = compute_array_steering_vector(tx_array.elements_pos, k_vec, r, lambda);
a_rx = compute_array_steering_vector(rx_array.elements_pos, k_vec, r, lambda);

% 应用距离衰减和相位补偿
a_tx = apply_range_compensation(a_tx, r, lambda);
a_rx = apply_range_compensation(a_rx, r, lambda);

end

function a = compute_array_steering_vector(elements_pos, k_vec, r, lambda)
    num_elements = size(elements_pos, 1);
    a = zeros(num_elements, 1);
    
    % 计算阵列中心
    array_center = mean(elements_pos, 1);
    
    % 高精度相位计算
    for i = 1:num_elements
        % 相对位置计算
        rel_pos = elements_pos(i,:) - array_center;
        
        % 精确相位计算
        phase = compute_precise_phase(rel_pos, k_vec, lambda);
        
        % 使用复数指数
        a(i) = exp(1j * phase);
    end
    
    % 归一化
    a = a / (norm(a) + eps);
    
    % 数值稳定性检查
    if any(isnan(a)) || any(isinf(a))
        warning('导向矢量计算出现数值不稳定');
        a(isnan(a) | isinf(a)) = 1/sqrt(num_elements);
        a = a / (norm(a) + eps);
    end
end

function phase = compute_precise_phase(pos, k_vec, lambda)
    % 高精度相位计算
    phase_components = 2*pi/lambda * pos * k_vec;
    phase = sum(phase_components);
    
    % 相位限制在[-π, π]范围内
    phase = mod(phase + pi, 2*pi) - pi;
end

function a = apply_range_compensation(a, r, lambda)
    % 距离衰减补偿
    k = 2*pi/lambda;
    range_factor = 1/sqrt(max(r, 0.1));  % 避免除零
    phase_comp = exp(-1j * k * r);
    
    a = a * (range_factor * phase_comp);
    
    % 再次归一化
    a = a / (norm(a) + eps);
end
