%% 太赫兹波束对准感知部分主函数
% 作者：
% 日期：

% 内存优化设置
maxNumCompThreads(2);  % 限制线程数
% feature('numCores',2); % 此行在某些MATLAB版本中不支持，注释掉

close all;
clear;
clc;

% 设置系统内存限制（单位：GB）
mem_limit = 4;
try
    limit_str = sprintf('memory limit %dGB', mem_limit);
    system(['setx MATLAB_MEMORY_LIMIT "' limit_str '"']);
catch
    fprintf('设置内存限制失败，继续执行程序\n');
end

%% 参数设置
params = set_parameters();

% 内存优化：减少采样点和chirp数
% 确保采样点数足够覆盖扫频时间
min_required_samples = ceil(params.fmcw.T * params.fmcw.fs) + 1;
params.fmcw.Ns = max(min_required_samples, params.fmcw.Ns);  % 确保采样点数足够
params.fmcw.num_chirps = min(params.fmcw.num_chirps, 64);  % 进一步限制chirp数量

% 设置跟踪模式：1-使用UKF，2-使用标准卡尔曼滤波器
track_mode = 1;  % 默认使用UKF滤波器

% 打印关键参数
fprintf('FMCW参数: T=%.3f ms, B=%.2f GHz, fs=%.2f MHz, 采样点数=%d\n', ...
    params.fmcw.T*1e3, params.fmcw.B/1e9, params.fmcw.fs/1e6, params.fmcw.Ns);
fprintf('阵列配置: 发射端 %dx%d, 接收端 %dx%d\n', ...
    params.tx.array_size(1), params.tx.array_size(2), ...
    params.rx.array_size(1), params.rx.array_size(2));
fprintf('跟踪模式: %s\n', conditional(track_mode == 1, '无迹卡尔曼滤波器(UKF)', '标准卡尔曼滤波器'));

%% 初始化发射接收阵列
try
    [tx_array, rx_array] = init_arrays(params);
    fprintf('阵列初始化成功: 发射端 %d 个阵元, 接收端 %d 个阵元\n', ...
        tx_array.num_elements, rx_array.num_elements);
catch ME
    fprintf('阵列初始化错误: %s\n', ME.message);
    rethrow(ME);
end

%% 主循环 - 模拟接收端移动并进行感知
num_frames = params.sim.num_frames;
results = struct('time', cell(1, num_frames), 'true', cell(1, num_frames), 'est', cell(1, num_frames));

% 初始化卡尔曼滤波器
if track_mode == 1
    % 使用无迹卡尔曼滤波器(UKF)替代传统卡尔曼滤波器
    % 初始状态向量 [r, vr, ar, az, vaz, aaz, el, vel, ael]
    % 从params中获取初始值
    r0 = norm(params.rx.init_pos);  % 初始距离
    vr0 = norm(params.rx.velocity);  % 初始径向速度
    az0 = atan2d(params.rx.init_pos(2), params.rx.init_pos(1));  % 初始方位角
    vaz0 = 0;  % 初始方位角速度
    el0 = atan2d(params.rx.init_pos(3), sqrt(params.rx.init_pos(1)^2 + params.rx.init_pos(2)^2));  % 初始俯仰角
    vel0 = 0;  % 初始俯仰角速度
    
    initial_state = [r0; vr0; 0; az0; vaz0; 0; el0; vel0; 0];
    
    % 设置测量噪声协方差
    % 测量噪声在实际应用中需要根据雷达性能确定
    R_base = diag([0.2^2, 0.5^2, 0.5^2]);  % 距离、方位角、俯仰角的测量噪声
    
    % 设置过程噪声协方差
    % 过程噪声反映状态演化的不确定性
    Q_base = diag([0.1^2, 1.0^2, 0.5^2, 0.2^2, 0.1^2, 0.05^2, 0.2^2, 0.1^2, 0.05^2]);
    
    % 使用UKF初始化函数
    kf = init_ukf_filter(initial_state, R_base, Q_base, params.sim.frame_interval);
    
    disp(['卡尔曼滤波器初始化: 初始状态 [r=' num2str(r0,'%.2f') ...
         ', vr=' num2str(vr0,'%.2f') ...
         ', ar=0.00, az=' num2str(az0,'%.2f') ...
         ', vaz=' num2str(vaz0,'%.2f') ', aaz=0.00, el=' num2str(el0,'%.2f') ...
         ', vel=' num2str(vel0,'%.2f') ', ael=0.00]']);
else
    % 初始化标准卡尔曼滤波器
    kf = init_kalman_filter(params);
end

% 接收端初始位置和速度 - 使用结构体存储完整运动状态
target = struct();
target.pos = params.rx.init_pos;
target.vel = params.rx.velocity;
target.acc = [0.5, -0.2, 0.3];  % 初始加速度 [ax, ay, az] (m/s^2)
target.jerk = [0.05, -0.04, 0.03];  % 加加速度 [jx, jy, jz] (m/s^3)
target.angular_vel = [0, 0, 0];  % 角速度 [ωx, ωy, ωz] (rad/s)
target.angular_acc = [0.001, 0.002, -0.001];  % 角加速度 [αx, αy, αz] (rad/s^2)

% 为轨迹模拟预先定义路径点或转向点
target.waypoints = 2;  % 设置轨迹中的变化点数量
target.waypoint_times = [num_frames * 0.3, num_frames * 0.7];  % 在这些帧处改变运动特性
target.phase = 1;  % 当前运动阶段

% 设置计时器以估计剩余时间
total_timer = tic;
frame_times = zeros(1, num_frames);

for frame_idx = 1:num_frames
    frame_timer = tic;
    fprintf('\n处理帧 %d/%d...\n', frame_idx, num_frames);
    
    % 检查是否到达转向点
    time = (frame_idx-1) * params.sim.frame_interval;
    if target.phase < target.waypoints && frame_idx >= target.waypoint_times(target.phase)
        target.phase = target.phase + 1;
        
        % 根据不同阶段设置不同的加速度
        if target.phase == 2
            % 第二阶段：急加速和转向
            target.acc = [-0.8, 1.2, 0.9];
            target.jerk = [0.1, 0.1, -0.05];
            target.angular_vel = [0.02, 0.03, -0.01];
            fprintf('运动模式变化: 进入快速机动阶段\n');
        else
            % 第三阶段：减速
            target.acc = [0.2, -0.7, -0.3];
            target.jerk = [-0.15, -0.12, 0.08];
            target.angular_vel = [-0.01, -0.02, 0.01];
            fprintf('运动模式变化: 进入减速阶段\n');
        end
    end
    
    % 更新加速度（随时间轻微变化，模拟真实场景）
    target.acc = target.acc + target.jerk * params.sim.frame_interval;
    
    % 更新速度（考虑加速度）
    target.vel = target.vel + target.acc * params.sim.frame_interval;
    
    % 限制最大速度（真实物理约束）
    max_speed = 25.0; % m/s
    vel_mag = norm(target.vel);
    if vel_mag > max_speed
        target.vel = target.vel * (max_speed / vel_mag);
    end
    
    % 更新位置（考虑当前速度和加速度）
    delta_pos = target.vel * params.sim.frame_interval + 0.5 * target.acc * params.sim.frame_interval^2;
    target.pos = target.pos + delta_pos;
    
    % 获取当前位置
    rx_pos = target.pos;
    
    % 计算接收端相对于发射端的球坐标
    % 计算水平距离和总距离
    horizontal_distance = sqrt(rx_pos(1)^2 + rx_pos(2)^2);
    range = sqrt(rx_pos(1)^2 + rx_pos(2)^2 + rx_pos(3)^2);
    
    % 计算方位角 (水平面内从x轴顺时针方向的角度)
    azimuth = atan2d(rx_pos(2), rx_pos(1));
    
    % 计算俯仰角 (从水平面到目标的角度，向上为正)
    elevation = atan2d(rx_pos(3), horizontal_distance);
    
    fprintf('真实位置: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', range, azimuth, elevation);
    fprintf('真实速度: vx=%.2f m/s, vy=%.2f m/s, vz=%.2f m/s\n', target.vel(1), target.vel(2), target.vel(3));
    fprintf('真实加速度: ax=%.2f m/s², ay=%.2f m/s², az=%.2f m/s²\n', target.acc(1), target.acc(2), target.acc(3));
    
    % 更新接收端阵列位置
    rx_array = update_rx_array(rx_array, rx_pos);
    
    % 对前3帧的特殊处理
    if frame_idx <= 3
        % 前3帧使用真实值初始化系统
        fprintf('初始化阶段(帧%d/3): 使用真实值初始化跟踪系统\n', frame_idx);
        measurement = [range; azimuth; elevation];
        
        % 真实速度的径向分量计算
        vr_true = (target.vel(1)*rx_pos(1) + target.vel(2)*rx_pos(2) + target.vel(3)*rx_pos(3)) / range;
        
        % 计算真实的方位角速度和俯仰角速度 (简化计算)
        vaz_true = 0.1;  % 简化值
        vel_true = 0.1;  % 简化值
        
        % 使用真实值更新UKF
        kf.x = [range; vr_true; 0; azimuth; vaz_true; 0; elevation; vel_true; 0];
        
        % 适度增加过程噪声，以便更好地适应真实运动
        kf.params.Q = kf.params.Q * 1.1;
        
        % 降低测量噪声，提高真实值的权重
        kf.params.R = kf.params.R * 0.4;
        
        % 直接存储当前状态为结果
        estimated_range = range;
        estimated_velocity = vr_true;
        estimated_acceleration = 0;
        estimated_azimuth = azimuth;
        estimated_vaz = vaz_true;
        estimated_aaz = 0;
        estimated_elevation = elevation;
        estimated_vel = vel_true;
        estimated_ael = 0;
        updated_state = kf.x;
        innovation = zeros(3, 1);
        NIS = 0;
    else
        % 正常的预测和更新过程
        % 预测阶段 - 使用上一帧的结果预测当前帧的位置
        dt = params.sim.frame_interval;
        [predicted_state, predicted_P, ukf_params] = ukf_predict(kf.x, kf.P, kf.params, dt);
        
        % 使用卡尔曼滤波预测作为先验信息
        prior_info.range = predicted_state(1);
        prior_info.azimuth = predicted_state(4);
        prior_info.elevation = predicted_state(7);
        
        % 计算预测协方差以设置搜索范围
        range_var = predicted_P(1,1);
        azimuth_var = predicted_P(4,4);
        elevation_var = predicted_P(7,7);
        
        % 设置搜索范围为预测值周围3个标准差
        prior_cov = diag([range_var, azimuth_var, elevation_var]);
        
        fprintf('卡尔曼滤波预测: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', prior_info.range, prior_info.azimuth, prior_info.elevation);
        
        % 生成发射信号
        tx_signal = generate_fmcw(params);
        
        % 模拟传播并得到接收信号
        rx_signal = simulate_propagation(tx_signal, tx_array, rx_array, params);
        
        % 内存优化：如果接收信号过大，进行下采样处理
        [signal_rows, signal_cols] = size(rx_signal);
        if signal_rows > 10000
            % 计算下采样因子，保持信号特性的同时减少数据量
            downsample_factor = ceil(signal_rows / 10000);
            rx_signal = rx_signal(1:downsample_factor:end, :);
            fprintf('内存优化: 接收信号下采样 %dx (从 %d 行到 %d 行)\n', ...
                downsample_factor, signal_rows, size(rx_signal, 1));
        end
        
        % 距离多普勒处理
        [rd_cube, range_axis, velocity_axis] = range_doppler_processing(rx_signal, params);
        
        % 为OMP算法准备额外参数
        params_omp = params;
        params_omp.omp.max_iter = 3;  % 减少迭代次数以提高速度
        
        % CFAR检测，获取距离和速度
        [detected_range, detected_velocity] = cfar_detection(rd_cube, range_axis, velocity_axis, params, prior_info.range);
        
        % 使用CFAR结果作为先验距离
        if ~isempty(detected_range)
            % 将检测到的单个距离值赋给先验
            prior_info.range = detected_range;
            
            % 选取与当前处理峰值对应的采样点范围
            peak_idx = round(prior_info.range * 2 * params.fmcw.mu / params.c * params.fmcw.fs);
            peak_idx = max(1, min(peak_idx, size(rx_signal, 1)-1));  % 确保索引有效
            
            % 保存CFAR检测结果到参数中，供OMP使用
            params_omp.detection = struct('range', detected_range, 'velocity', detected_velocity);
            
            fprintf('CFAR检测: 距离=%.2f m, 速度=%.2f m/s\n', detected_range, detected_velocity);
        else
            fprintf('CFAR检测: 未检测到有效目标\n');
            params_omp.detection = struct('range', [], 'velocity', []);
        end
        
        % OMP稀疏重建 - 结合优化的先验信息
        [est_range, est_azimuth, est_elevation] = prior_guided_omp(rx_signal, tx_array, rx_array, prior_info, prior_cov, params_omp);
        fprintf('OMP估计结果: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°\n', est_range, est_azimuth, est_elevation);
        
        % 检查OMP结果与预测值之间的差异，如过大则降低权重
        range_diff = abs(est_range - prior_info.range);
        az_diff = abs(wrapTo180(est_azimuth - prior_info.azimuth));
        el_diff = abs(wrapTo180(est_elevation - prior_info.elevation));
        
        % 根据实际测量情况设置更有针对性的阈值
        range_threshold = max(1.5, 0.08 * prior_info.range);  % 更严格的动态距离阈值
        az_threshold = 5.0;  % 降低方位角阈值
        el_threshold = 5.0;  % 降低俯仰角阈值
        
        % 设置异常检测标志
        measurement_anomaly = false;
        
        % 检查并处理距离异常
        if range_diff > range_threshold
            measurement_anomaly = true;
            fprintf('警告: 距离估计与预测差异过大(%.2f > %.2f)，调整处理\n', range_diff, range_threshold);
            
            % 检查是否可能使用CFAR结果代替OMP估计
            if ~isempty(detected_range) && abs(detected_range - prior_info.range) < range_diff
                fprintf('使用CFAR距离(%.2f)替代OMP距离(%.2f)\n', detected_range, est_range);
                est_range = detected_range;
            else
                % 没有可用的CFAR结果时，使用加权平均
                weight = 0.75;  % 更偏向预测值
                est_range = weight * prior_info.range + (1-weight) * est_range;
                fprintf('应用距离加权平均校正: %.2f\n', est_range);
            end
        end
        
        % 检查并处理角度异常
        if az_diff > az_threshold
            measurement_anomaly = true;
            fprintf('警告: 方位角估计与预测差异过大(%.2f > %.2f)，调整处理\n', az_diff, az_threshold);

            % 方位角异常处理 - 自适应融合策略
            az_trust_factor = max(0.2, min(0.5, 1.0 - az_diff/20.0));  % 自适应信任因子
            weight = 1.0 - az_trust_factor;  % 预测值权重
            est_azimuth = weight * prior_info.azimuth + az_trust_factor * est_azimuth;
            fprintf('应用方位角自适应融合: %.2f (预测×%.2f + 测量×%.2f)\n', ...
                    est_azimuth, weight, az_trust_factor);
        end
        
        if el_diff > el_threshold
            measurement_anomaly = true;
            fprintf('警告: 俯仰角估计与预测差异过大(%.2f > %.2f)，调整处理\n', el_diff, el_threshold);
            
            % 俯仰角异常处理 - 自适应融合策略
            el_trust_factor = max(0.2, min(0.5, 1.0 - el_diff/20.0));  % 自适应信任因子
            weight = 1.0 - el_trust_factor;  % 预测值权重
            est_elevation = weight * prior_info.elevation + el_trust_factor * est_elevation;
            fprintf('应用俯仰角自适应融合: %.2f (预测×%.2f + 测量×%.2f)\n', ...
                    est_elevation, weight, el_trust_factor);
        end
        
        % 调整测量噪声
        if measurement_anomaly
            % 创建临时测量噪声矩阵，根据异常程度动态调整
            temp_R = kf.params.R .* diag([
                1 + min(1.5, range_diff/range_threshold), ...
                1 + min(2.0, az_diff/az_threshold), ...
                1 + min(2.0, el_diff/el_threshold)
            ]);
            
            fprintf('调整测量噪声增益: [%.2f, %.2f, %.2f]\n', ...
                temp_R(1,1)/kf.params.R(1,1), ...
                temp_R(2,2)/kf.params.R(2,2), ...
                temp_R(3,3)/kf.params.R(3,3));
        else
            % 正常测量，进一步降低噪声以提高测量权重
            temp_R = kf.params.R .* diag([
                max(0.7, min(1.1, range_diff/(0.3*range_threshold))), ...
                max(0.7, min(1.1, az_diff/(0.3*az_threshold))), ...
                max(0.7, min(1.1, el_diff/(0.3*el_threshold)))
            ]);
            
            % 如果测量与预测极为接近，进一步增强测量权重
            if range_diff < 0.2 * range_threshold && ...
               az_diff < 0.2 * az_threshold && ...
               el_diff < 0.2 * el_threshold
                temp_R = temp_R * 0.7;  % 显著降低测量噪声
                fprintf('测量与预测极为接近，显著增强测量权重\n');
            end
        end
        
        % 卡尔曼滤波更新 - 将测量结果与预测结果进行融合
        % 更新测量值，使用可能经过修正的估计值
        measurement = [est_range; est_azimuth; est_elevation];
        
        % 保存原始噪声矩阵以便恢复
        original_R = kf.params.R;
        % 为UKF更新传递调整后的噪声协方差
        kf.params.R = temp_R;
        
        % 添加异常处理，防止单个帧的错误导致整个程序崩溃
        try
            % 状态更新前检查预测状态是否有异常
            if any(isnan(predicted_state)) || any(isinf(predicted_state))
                error('预测状态包含NaN或Inf值，可能需要重新初始化滤波器');
            end
            
            % 使用更新后的测量值更新状态
            [updated_state, updated_P, ukf_params, NIS] = ukf_update(predicted_state, predicted_P, measurement, ukf_params);
            
            % NIS异常值检测
            if NIS > 20.0  % 创新序列异常大
                fprintf('警告: 创新序列异常大(NIS=%.2f)，可能表明滤波器发散\n', NIS);
                
                % 计算一个置信度因子，根据NIS大小调整状态更新
                confidence = min(1.0, max(0.3, 20.0/NIS));
                
                % 调整更新：更新状态 = 置信度 * 更新状态 + (1-置信度) * 预测状态
                corrected_state = confidence * updated_state + (1-confidence) * predicted_state;
                
                % 报告调整
                fprintf('应用状态校正，置信度=%.2f\n', confidence);
                
                % 保存调整后的状态
                updated_state = corrected_state;
                end
                
            % 应用物理约束以确保估计的合理性
            % 确保距离非负
            updated_state(1) = max(0.1, updated_state(1));
            
            % 限制角度在有效范围内
            updated_state(4) = wrapTo180(updated_state(4));  % 方位角
            updated_state(7) = wrapTo180(updated_state(7));  % 俯仰角
            
            % 限制速度和加速度在合理范围内
            max_vr = 40.0;  % 最大径向速度
            max_ar = 20.0;  % 最大径向加速度
            
            if abs(updated_state(2)) > max_vr
                updated_state(2) = sign(updated_state(2)) * max_vr;
            end
            
            if abs(updated_state(3)) > max_ar
                updated_state(3) = sign(updated_state(3)) * max_ar;
            end
            
            % 提取最终估计
            estimated_range = updated_state(1);
            estimated_velocity = updated_state(2);
            estimated_acceleration = updated_state(3);
            estimated_azimuth = updated_state(4);
            estimated_vaz = updated_state(5);
            estimated_aaz = updated_state(6);
            estimated_elevation = updated_state(7);
            estimated_vel = updated_state(8);
            estimated_ael = updated_state(9);
            
            % 更新kf结构体
            kf.x = updated_state;
            kf.P = updated_P;
            kf.params = ukf_params;
            
            % 恢复原始噪声矩阵，避免累积影响
            kf.params.R = original_R;
            
            % 记录创新序列用于显示
            innovation = measurement - [updated_state(1); updated_state(4); updated_state(7)];
            % 处理角度的周期性
            innovation(2) = wrapTo180(innovation(2));
            innovation(3) = wrapTo180(innovation(3));
            
        catch ME
            % 发生错误时，回退到预测状态
            fprintf('错误发生在帧 %d 的更新阶段: %s\n', frame_idx, ME.message);
            fprintf('回退到预测状态以保证系统稳定性\n');
            
            % 使用预测状态作为最终状态
            updated_state = predicted_state;
            estimated_range = predicted_state(1);
            estimated_velocity = predicted_state(2);
            estimated_acceleration = predicted_state(3);
            estimated_azimuth = predicted_state(4);
            estimated_vaz = predicted_state(5);
            estimated_aaz = predicted_state(6);
            estimated_elevation = predicted_state(7);
            estimated_vel = predicted_state(8);
            estimated_ael = predicted_state(9);
            
            % 设置虚拟创新序列
            innovation = [0; 0; 0];
            NIS = 0;
            
            % 恢复原始噪声矩阵
            kf.params.R = original_R;
        end
    end
    
    % 显示创新序列和马氏距离(NIS)
    fprintf('创新序列: 距离=%.2f m, 方位角=%.2f°, 俯仰角=%.2f°, NIS=%.2f\n', ...
        innovation(1), innovation(2), innovation(3), NIS);
    
    % 显示状态估计
    fprintf('状态估计: r=%.2f m, vr=%.2f m/s, ar=%.2f m/s²\n', ...
        estimated_range, estimated_velocity, estimated_acceleration);
    fprintf('角度估计: az=%.2f°(v=%.2f°/s, a=%.2f°/s²), el=%.2f°(v=%.2f°/s, a=%.2f°/s²)\n', ...
        estimated_azimuth, estimated_vaz, estimated_aaz, estimated_elevation, estimated_vel, estimated_ael);
    
    % 存储结果
    results(frame_idx).time = time;
    results(frame_idx).true = [range, azimuth, elevation];
    results(frame_idx).est = [estimated_range, estimated_azimuth, estimated_elevation];
    
    % 计算并显示当前帧误差
    range_error = estimated_range - range;
    az_error = estimated_azimuth - azimuth;
    el_error = estimated_elevation - elevation;
    
    fprintf('\n当前帧估计误差:\n');
    fprintf('距离: 真实=%.2f m, 估计=%.2f m, 误差=%.2f m, 比例=%.2f\n', ...
            range, estimated_range, range_error, range_error/range);
    fprintf('方位角: 真实=%.2f°, 估计=%.2f°, 误差=%.2f°\n', ...
            azimuth, estimated_azimuth, az_error);
    fprintf('俯仰角: 真实=%.2f°, 估计=%.2f°, 误差=%.2f°\n', ...
            elevation, estimated_elevation, el_error);
        
    % 可视化当前结果
    if mod(frame_idx, params.viz.update_interval) == 0
        visualize_tracking(results(1:frame_idx), params);
    end
    
    % 记录帧处理时间
    frame_times(frame_idx) = toc(frame_timer);
    mean_frame_time = mean(frame_times(1:frame_idx));
    est_remaining_time = mean_frame_time * (num_frames - frame_idx);
    
    fprintf('帧处理时间: %.2f 秒, 估计剩余时间: %.1f 分钟\n', ...
        frame_times(frame_idx), est_remaining_time/60);
end

total_time = toc(total_timer);
fprintf('\n处理完成。总耗时: %.1f 分钟\n', total_time/60);

%% 结果评估
try
    evaluate_performance(results, params);
catch ME
    fprintf('结果评估错误: %s\n', ME.message);
end

% 定义辅助函数，用于条件选择
function result = conditional(condition, true_value, false_value)
    if condition
        result = true_value;
    else
        result = false_value;
    end
end
