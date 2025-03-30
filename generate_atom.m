function atom = generate_atom(a_tx, a_rx, range, params)
%GENERATE_ATOM 高精度字典原子生成
%   实现高精度的FMCW信号模型，包含精确的多普勒效应处理

% 提取参数
c = params.c;                    % 光速
fc = params.fc;                  % 载波频率
lambda = c / fc;                 % 波长
fs = params.fmcw.fs;            % 采样率
sweep_rate = params.fmcw.mu;     % 调频率
T = params.fmcw.T;              % 扫频周期

% 获取当前信号长度
if isfield(params, 'current_signal_length')
    signal_length = params.current_signal_length;
else
    signal_length = params.fmcw.Ns;
end

% 精确时延计算
tau = 2 * range / c;

% 计算拍频
beat_freq = sweep_rate * tau;

% 高精度时间向量
t = (0:signal_length-1)' / fs;

% 完整的FMCW信号相位计算
% 1. 载波相位
carrier_phase = 2*pi * fc * tau;

% 2. 调频引起的相位
chirp_phase = pi * sweep_rate * tau^2;

% 3. 拍频相位
beat_phase = 2*pi * beat_freq * t;

% 4. 计算径向速度和多普勒效应
doppler_phase = zeros(size(t));
if isfield(params, 'target_state')
    % 从目标状态获取速度信息
    target_vel = params.target_state.velocity;
    target_pos = params.target_state.position;
    
    % 计算径向速度
    if isfield(params, 'platform_state')
        % 考虑平台运动
        platform_pos = params.platform_state.position;
        platform_vel = params.platform_state.velocity;
        
        % 计算相对位置和速度
        rel_pos = target_pos - platform_pos;
        rel_vel = target_vel - platform_vel;
    else
        % 只考虑目标运动
        rel_pos = target_pos;
        rel_vel = target_vel;
    end
    
    % 计算径向单位向量
    r_unit = rel_pos / (norm(rel_pos) + eps);
    
    % 计算径向速度
    v_r = dot(rel_vel, r_unit);
    
    % 计算多普勒频率
    doppler_freq = 2 * v_r / lambda;
    
    % 计算多普勒相位
    doppler_phase = 2*pi * doppler_freq * t;
    
    % 添加二阶多普勒效应（如果有加速度信息）
    if isfield(params.target_state, 'acceleration')
        target_acc = params.target_state.acceleration;
        if isfield(params, 'platform_state') && isfield(params.platform_state, 'acceleration')
            platform_acc = params.platform_state.acceleration;
            rel_acc = target_acc - platform_acc;
        else
            rel_acc = target_acc;
        end
        
        % 计算径向加速度
        a_r = dot(rel_acc, r_unit);
        
        % 添加加速度引起的相位变化
        doppler_phase = doppler_phase + pi * (2/lambda) * a_r * t.^2;
    end
end

% 总相位
phase = carrier_phase + chirp_phase - beat_phase + doppler_phase;

% 生成基础信号
atom_base = exp(1j * phase);

% 应用天线阵列因子，考虑动态空间特性
if isfield(params, 'dynamic_array') && params.dynamic_array.enabled
    array_factor = compute_dynamic_array_factor(a_tx, a_rx, range, params);
else
    array_factor = (a_rx' * a_tx);
end

% 生成完整原子
atom = atom_base * array_factor;

% 添加距离衰减和大气衰减
range_attenuation = compute_range_attenuation(range, params);
atom = atom * range_attenuation;

% 信号归一化
norm_factor = 1 / sqrt(length(atom_base));
atom = atom * norm_factor;

% 确保列向量格式
if size(atom, 2) > size(atom, 1)
    atom = atom';
end

% 验证信号长度并调整
if length(atom) ~= signal_length
    if length(atom) > signal_length
        atom = atom(1:signal_length);
    else
        atom = [atom; zeros(signal_length-length(atom), 1)];
    end
end

% 数值稳定性检查和处理
atom = ensure_numerical_stability(atom);

end

function array_factor = compute_dynamic_array_factor(a_tx, a_rx, range, params)
    % 计算考虑动态效应的阵列因子
    array_factor = (a_rx' * a_tx);
    
    if range > params.dynamic_array.range_threshold
        % 远场条件下的修正
        array_factor = array_factor * exp(-1j * pi/4) / sqrt(range);
    end
end

function attenuation = compute_range_attenuation(range, params)
    % 基础自由空间损耗
    basic_loss = 1 / (range^2 + eps);
    
    % 如果配置了大气衰减参数
    if isfield(params, 'atmosphere')
        % 添加大气衰减
        alpha_db_per_km = params.atmosphere.attenuation_db_per_km;
        alpha_linear = 10^(-alpha_db_per_km/10);
        atmosphere_loss = alpha_linear^(range/1000);
        attenuation = sqrt(basic_loss * atmosphere_loss);
    else
        attenuation = sqrt(basic_loss);
    end
end

function atom = ensure_numerical_stability(atom)
    % 处理数值不稳定
    if any(isnan(atom)) || any(isinf(atom))
        warning('原子生成出现数值不稳定');
        atom(isnan(atom) | isinf(atom)) = eps;
    end
    
    % 确保最小幅度
    min_amp = 1e-10;
    small_indices = abs(atom) < min_amp;
    if any(small_indices)
        atom(small_indices) = min_amp * exp(1j * angle(atom(small_indices)));
    end
    
    % 最终归一化
    atom = atom / (norm(atom) + eps);
end
