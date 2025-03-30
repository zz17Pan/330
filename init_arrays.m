function [tx_array, rx_array] = init_arrays(params)
%INIT_ARRAYS 初始化发射和接收天线阵列
%   params: 系统参数结构体
%   tx_array: 发射阵列结构体
%   rx_array: 接收阵列结构体

% 发射阵列初始化
tx_array = struct();
tx_array.size = params.tx.array_size;
tx_array.pos = params.tx.pos;
tx_array.spacing = params.tx.spacing;

% 初始化发射阵列天线元素坐标
[tx_array.elements_pos, tx_array.num_elements] = create_array_elements(tx_array.size, tx_array.spacing, tx_array.pos);

% 接收阵列初始化
rx_array = struct();
rx_array.size = params.rx.array_size;
rx_array.pos = params.rx.init_pos;
rx_array.spacing = params.rx.spacing;

% 初始化接收阵列天线元素坐标
[rx_array.elements_pos, rx_array.num_elements] = create_array_elements(rx_array.size, rx_array.spacing, rx_array.pos);

% 打印接收阵列信息以便调试
fprintf('接收阵列初始位置: [%.2f, %.2f, %.2f]\n', rx_array.pos(1), rx_array.pos(2), rx_array.pos(3));
fprintf('接收阵列元素数量: %d\n', rx_array.num_elements);
fprintf('接收阵列第一个元素位置: [%.2f, %.2f, %.2f]\n', rx_array.elements_pos(1,1), rx_array.elements_pos(1,2), rx_array.elements_pos(1,3));
fprintf('接收阵列最后一个元素位置: [%.2f, %.2f, %.2f]\n', rx_array.elements_pos(end,1), rx_array.elements_pos(end,2), rx_array.elements_pos(end,3));

end

function [elements_pos, num_elements] = create_array_elements(array_size, spacing, array_pos)
%CREATE_ARRAY_ELEMENTS 创建阵列中各天线元素的坐标
%   array_size: 阵列大小 [行, 列]
%   spacing: 阵元间距
%   array_pos: 阵列中心位置 [x, y, z]
%   elements_pos: 各天线元素的坐标 (Nx3矩阵, 每行为一个天线元素的[x,y,z]坐标)
%   num_elements: 天线元素总数

% 计算天线元素总数
num_rows = array_size(1);
num_cols = array_size(2);
num_elements = num_rows * num_cols;

% 初始化元素坐标矩阵
elements_pos = zeros(num_elements, 3);

% 计算每个天线元素的坐标
element_idx = 1;
for row = 1:num_rows
    for col = 1:num_cols
        % 计算相对于阵列中心的偏移
        % 确保阵列中心是(0,0,0)，且阵列在xy平面上，z=0
        x_offset = (col - (num_cols+1)/2) * spacing;
        y_offset = (row - (num_rows+1)/2) * spacing;
        z_offset = 0; % 阵列在xy平面上
        
        % 计算全局坐标
        elements_pos(element_idx, :) = array_pos + [x_offset, y_offset, z_offset];
        element_idx = element_idx + 1;
    end
end

end 