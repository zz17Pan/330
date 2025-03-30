function [a_tx, a_rx] = compute_steering_vector(tx_array, rx_array, r, az, el, params)
%COMPUTE_STEERING_VECTOR 高精度导向矢量计算
%   实现亚度级角度精度的导向矢量计算
%   r: 距离 (m)
%   az: 方位角 (度)
%   el: 俯仰角 (度)

% 参数预处理和检查
if r < 0.1
    r = 0.1;
end

% 提取基本参数
c = params.c;           % 光速
fc = params.fc;        % 载波频率
lambda = c / fc;       % 波长
k = 2*pi/lambda;       % 波数

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
    k_dir = k_dir + eps;
    k_norm = norm(k_dir);
end
k_dir = k_dir / k_norm;

% 计算发射阵列导向矢量
a_tx = compute_array_vector(tx_array.elements_pos, k_dir, k, r, lambda);

% 计算接收阵列导向矢量
a_rx = compute_array_vector(rx_array.elements_pos, k_dir, k, r, lambda);

% 应用相位补偿
a_tx = apply_phase_compensation(a_tx, r, k);
a_rx = apply_phase_compensation(a_rx, r, k);

% 数值稳定性检查
a_tx = ensure_numerical_stability(a_tx);
a_rx = ensure_numerical_stability(a_rx);
end

function a = compute_array_vector(elements_pos, k_dir, k, r, lambda)
    % 获取阵元数量
    num_elements = size(elements_pos, 1);
    
    % 计算阵列中心
    array_center = mean(elements_pos, 1);
    
    % 初始化导向矢量
    a = zeros(num_elements, 1);
    
    % 对每个阵元计算相位
    for i = 1:num_elements
        % 计算相对位置
        rel_pos = elements_pos(i,:) - array_center;
        
        % 计算精确相位
        phase = compute_precise_phase(rel_pos, k_dir, k, r, lambda);
        
        % 生成导向矢量元素
        a(i) = exp(1j * phase);
    end
    
    % 归一化
    a = a / (norm(a) + eps);
end

function phase = compute_precise_phase(pos, k_dir, k, r, lambda)
    % 计算空间相位
    spatial_phase = k * (pos * k_dir);
    
    % 计算传播相位
    prop_phase = -k * r;
    
    % 计算总相位
    phase = spatial_phase + prop_phase;
    
    % 相位限制在[-π, π]范围内
    phase = mod(phase + pi, 2*pi) - pi;
    
    % 添加小量以避免数值不稳定
    if abs(phase) < eps
        phase = phase + eps;
    end
end

function a = apply_phase_compensation(a, r, k)
    % 距离衰减补偿
    r = max(r, 0.1);  % 避免零距离
    amplitude = 1/sqrt(r);
    
    % 相位补偿
    phase_comp = exp(-1j * k * r);
    
    % 应用补偿
    a = a * (amplitude * phase_comp);
    
    % 再次归一化
    a = a / (norm(a) + eps);
end

function a = ensure_numerical_stability(a)
    % 检查数值稳定性
    if any(isnan(a)) || any(isinf(a))
        warning('导向矢量计算出现数值不稳定');
        % 替换无效值
        a(isnan(a) | isinf(a)) = eps;
        % 重新归一化
        a = a / (norm(a) + eps);
    end
    
    % 确保最小幅度
    min_amp = 1e-10;
    small_indices = abs(a) < min_amp;
    if any(small_indices)
        a(small_indices) = min_amp * exp(1j * angle(a(small_indices)));
        a = a / (norm(a) + eps);
    end
end
