function rx_array = update_rx_array(rx_array, new_pos)
%UPDATE_RX_ARRAY 更新接收阵列位置
%   rx_array: 接收阵列结构体
%   new_pos: 新的阵列中心位置 [x, y, z]
%   返回更新后的接收阵列结构体

% 计算位置偏移量
pos_offset = new_pos - rx_array.pos;

% 更新阵列中心位置
rx_array.pos = new_pos;

% 更新每个天线元素的位置
for i = 1:rx_array.num_elements
    rx_array.elements_pos(i, :) = rx_array.elements_pos(i, :) + pos_offset;
end

end 