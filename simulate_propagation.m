function rx_signal = simulate_propagation(tx_signal, tx_array, rx_array, params)
%SIMULATE_PROPAGATION 模拟信号从发射端传播到接收端的过程
%   tx_signal: 发射信号矩阵 [采样点数 x chirp数]
%   tx_array: 发射阵列结构体
%   rx_array: 接收阵列结构体
%   params: 系统参数结构体
%   rx_signal: 接收信号矩阵 [采样点数 x chirp数 x 接收阵元数]

% 提取参数
c = params.c;               % 光速
lambda = params.lambda;     % 波长
fc = params.fc;             % 载波频率
T = params.fmcw.T;          % 扫频时间
fs = params.fmcw.fs;        % 采样率
Ns = params.fmcw.Ns;        % 每个chirp的采样点数
num_chirps = params.fmcw.num_chirps;  % chirp数量
sweep_rate = params.fmcw.mu;  % 调频率

% 检查并可能降低采样点数
if Ns > 6000
    % 限制最大下采样因子，保证不过度降采样
    max_down_factor = 2; 
    down_factor = min(max_down_factor, ceil(Ns / 6000));
    Ns_reduced = ceil(Ns / down_factor);
    fprintf('内存优化: 传播模拟采用降采样，减少点数 %d -> %d\n', Ns, Ns_reduced);
    t = (0:(Ns_reduced-1))' * down_factor / fs;  % 降采样的时间向量
else
    Ns_reduced = Ns;
    t = (0:(Ns_reduced-1))' / fs;  % 原始时间向量
end

% 计算接收阵列相对于发射阵列的球坐标参数
rx_center = rx_array.pos;  % 接收阵列中心

% 计算水平距离和总距离
horizontal_distance = sqrt(rx_center(1)^2 + rx_center(2)^2);
r = sqrt(rx_center(1)^2 + rx_center(2)^2 + rx_center(3)^2);

% 计算方位角 (水平面内从x轴顺时针方向的角度)
azimuth = atan2d(rx_center(2), rx_center(1));

% 计算俯仰角 (从水平面到目标的角度，向上为正)
elevation = atan2d(rx_center(3), horizontal_distance);

% 打印详细的角度信息以便调试
fprintf('模拟信号传播 - 接收中心: [%.2f, %.2f, %.2f], 距离: %.2f m\n', rx_center(1), rx_center(2), rx_center(3), r);
fprintf('计算的方位角: %.2f°, 俯仰角: %.2f°\n', azimuth, elevation);

% 计算径向速度 (使用连续两帧的位置估计)
if isfield(rx_array, 'prev_pos')
    delta_t = params.sim.frame_interval;
    velocity = (rx_center - rx_array.prev_pos) / delta_t;
    % 径向速度是速度在发射端到接收端连线方向上的投影
    direction = rx_center / norm(rx_center);  % 单位方向向量
    radial_velocity = dot(velocity, direction);
else
    % 初始帧没有前一个位置，假设径向速度为0
    radial_velocity = 0;
end
rx_array.prev_pos = rx_center;  % 保存当前位置用于下一帧计算

% 计算多普勒频移
doppler_shift = 2 * radial_velocity * fc / c;

% 预计算一些值以加速处理
chirp_times = (0:num_chirps-1) * T;  % 每个chirp的时间偏移

% 将角度转为弧度进行计算
az_rad = deg2rad(azimuth);
el_rad = deg2rad(elevation);

% 计算波数向量 (与MUSIC中保持一致)
k_vector = 2*pi/lambda * [cos(el_rad)*cos(az_rad); 
                         cos(el_rad)*sin(az_rad); 
                         sin(el_rad)];

% 评估真实内存需求
num_rx_elements = rx_array.num_elements;
memory_per_element = Ns_reduced * num_chirps * 16 / 1024 / 1024;  % 假设复数为16字节，转为MB
total_memory = memory_per_element * num_rx_elements;

% 设置插值标志 - 默认不执行插值
if ~isfield(params.channel, 'interpolate_after_sampling')
    params.channel.interpolate_after_sampling = false;
end

% 使用常规处理方式
rx_signal = zeros(Ns_reduced, num_chirps, num_rx_elements);

% 内存优化: 每次处理一批接收阵元
batch_size = 4; % 每次处理4个接收阵元
num_batches = ceil(num_rx_elements / batch_size);

fprintf('内存优化: 分批处理接收阵元，每批%d个，共%d批\n', batch_size, num_batches);

for batch = 1:num_batches
    start_idx = (batch-1) * batch_size + 1;
    end_idx = min(batch * batch_size, num_rx_elements);
    fprintf('处理接收阵元批次 %d/%d (阵元 %d-%d)...\n', batch, num_batches, start_idx, end_idx);
    
    for rx_idx = start_idx:end_idx
        % 计算发射端到当前接收阵元的距离
        rx_element_pos = rx_array.elements_pos(rx_idx, :);
        distance = norm(rx_element_pos);
        
        % 计算时延
        tau = distance / c;
        
        % 时延是否在合理范围内
        if tau > T
            continue;  % 时延超过一个chirp的持续时间，跳过处理
        end
        
        % 计算相位偏移 (和接收阵元位置有关)
        pos_vector = rx_element_pos'; % 转为列向量
        phase_shift = k_vector' * pos_vector; % 空间相位
        
        % 计算时移后的基带信号
        time_shift = t - tau;
        
        % 只处理时延在采样窗口内的部分
        valid_samples = time_shift >= 0 & time_shift < T;
        
        % 如果没有有效样本，跳过
        if ~any(valid_samples)
            continue;
end

        % 计算延时信号 (基带信号) - 使用向量化操作优化
        delayed_signal = zeros(length(t), 1);
        delayed_signal(valid_samples) = exp(1j * 2 * pi * (0.5 * sweep_rate * time_shift(valid_samples).^2));
        
        % 对每个chirp计算接收信号
        for chirp_idx = 1:num_chirps
            % 当前chirp的时间偏移
            chirp_time = chirp_times(chirp_idx);
            
            % 计算当前chirp的多普勒相位
            chirp_doppler_phase = exp(1j * 2 * pi * doppler_shift * chirp_time);
            
            % 组合所有相位因子为一个复数标量，避免多次矩阵乘法
            combined_phase = chirp_doppler_phase * exp(1j * phase_shift);
            
            % 直接应用到信号上
            rx_signal(:, chirp_idx, rx_idx) = delayed_signal * combined_phase;
        end
    end
end

% 加入NLoS路径 (反射)
if params.channel.num_reflectors > 0
    % 简化反射路径，只保留少量强反射
    max_reflectors = min(params.channel.num_reflectors, 2); % 最多2个反射体
    
    for reflector_idx = 1:max_reflectors
        % 随机生成反射体位置，但限制在更合理的范围内
        max_reflector_range = min(20, norm(rx_array.pos)/4);  % 反射体最大距离
        reflector_pos = [(rand-0.5)*max_reflector_range, (rand-0.5)*max_reflector_range, max_reflector_range/2 + (rand-0.5)*max_reflector_range/4];
        
        % 反射路径的衰减系数 - 增加衰减以降低影响
        attenuation = params.channel.reflection_coef * 0.5;
        
        % 为该反射体处理所有接收阵元 - 批量处理
        for batch = 1:num_batches
            start_idx = (batch-1) * batch_size + 1;
            end_idx = min(batch * batch_size, num_rx_elements);
            
            for rx_idx = start_idx:end_idx
                % 计算反射路径的总距离
                rx_element_pos = rx_array.elements_pos(rx_idx, :);
                distance_tx_to_reflector = norm(reflector_pos);
                distance_reflector_to_rx = norm(reflector_pos - rx_element_pos);
                total_distance = distance_tx_to_reflector + distance_reflector_to_rx;
                
                % 计算反射路径时延
                tau_reflect = total_distance / c;
                
                % 跳过超出合理范围的时延
                if tau_reflect > T
                    continue;
                end
                
                % 简化处理：使用单一随机相位偏移
                phase_shift_reflect = 2*pi * rand;

                % 计算时移后的基带信号
                time_shift = t - tau_reflect;
                valid_samples = time_shift >= 0 & time_shift < T;
                
                if ~any(valid_samples)
                    continue;
end

                % 快速计算反射信号并应用于选定的chirp
                % 只处理部分chirp以减少计算负担
                for chirp_idx = 1:2:num_chirps
                    % 简化计算：使用单一多普勒相位
                    chirp_phase = exp(1j * (2 * pi * doppler_shift * chirp_times(chirp_idx) + phase_shift_reflect));
                    
                    % 计算当前反射的延时基带信号
                    delayed_signal = zeros(length(t), 1);
                    delayed_signal(valid_samples) = exp(1j * 2 * pi * (0.5 * sweep_rate * time_shift(valid_samples).^2));

                    % 将反射信号加到接收信号上
                    rx_signal(:, chirp_idx, rx_idx) = rx_signal(:, chirp_idx, rx_idx) + attenuation * delayed_signal * chirp_phase;
                end
                
                % 对于未处理的chirp，使用插值
                for chirp_idx = 2:2:num_chirps
                    if chirp_idx < num_chirps
                        rx_signal(:, chirp_idx, rx_idx) = rx_signal(:, chirp_idx, rx_idx) + 0.5 * (rx_signal(:, max(1, chirp_idx-1), rx_idx) + rx_signal(:, min(num_chirps, chirp_idx+1), rx_idx));
                    else
                        rx_signal(:, chirp_idx, rx_idx) = rx_signal(:, chirp_idx, rx_idx) + rx_signal(:, chirp_idx-1, rx_idx);
                    end
                end
            end
        end
    end
end

% 添加噪声 - 内存优化版本
if params.channel.add_noise
    % 计算信号功率（只采样部分点以节省内存）
    sample_size = min(1000, numel(rx_signal));
    sample_indices = randi(numel(rx_signal), sample_size, 1);
    sampled_signal = rx_signal(sample_indices);
    signal_power = mean(abs(sampled_signal).^2);
    
    if signal_power < 1e-10
        signal_power = 1e-10;
    end
    
    % 根据SNR计算噪声功率
    snr_linear = 10^(params.channel.snr/10);
    noise_power = signal_power / snr_linear;
        
    % 逐元素添加噪声，避免维度不匹配问题
    fprintf('添加噪声: SNR=%.1f dB\n', params.channel.snr);
    
    % 为每个接收阵元和每个chirp分别添加噪声
    for rx_idx = 1:num_rx_elements
        for chirp_idx = 1:num_chirps
            % 为当前信号生成噪声，确保维度匹配
            noise = sqrt(noise_power/2) * (randn(Ns_reduced, 1) + 1j*randn(Ns_reduced, 1));
        
            % 将噪声加入接收信号
            rx_signal(:, chirp_idx, rx_idx) = rx_signal(:, chirp_idx, rx_idx) + noise;
    end
    end
end

% 如果使用了降采样，需要插值回原始Ns（可选，取决于后续处理是否需要完整样本）
if Ns_reduced < Ns && params.channel.interpolate_after_sampling
    % 执行插值恢复为原始采样率
    fprintf('内存优化: 执行插值恢复为原始采样率 %d -> %d\n', Ns_reduced, Ns);
    rx_signal_full = zeros(Ns, num_chirps, num_rx_elements);
    
    % 使用线性插值恢复采样
    for rx_idx = 1:num_rx_elements
        for chirp_idx = 1:num_chirps
            % 线性插值到原始采样率
            rx_signal_full(:, chirp_idx, rx_idx) = interp1(t, rx_signal(:, chirp_idx, rx_idx), (0:Ns-1)/fs, 'linear', 0);
    end
    end
    rx_signal = rx_signal_full;
end

end 