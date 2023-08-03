function [time_vec, angle_vec] = decode_bfi(fname)
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

packets_num = size(val,1);

angle_arr = zeros(packets_num, 10);

%PHI11, 21, 31, 22, 32, PSI21, 31, 41, 32, 42
num_subcarrier = 52;
bphi = 6;
bpsi = 4;
angle_vec = zeros(packets_num, num_subcarrier,10);
per_subcarrier_bits = (5*bphi + 5*bpsi);
time_vec = zeros(packets_num, 1);
flag_use_order_protocol = true;
for i = 1:packets_num
    string_temp = val(i).x_source.layers.wlan_1.FixedParameters.wlan_vht_compressed_beamforming_report;
    time_vec(i) = str2num(val(i).x_source.layers.frame.frame_time_epoch);
    split_string = split(string_temp, ':');
    join_string = join(split_string(3:end),'');  % 参考Table 8-53f，以及下面段落的描述
    join_string = join_string{1};
    slength = length(join_string);
    bin_char_vec = [];
    for char_i = 1:slength
%         if char_i == slength - 1
%             continue; % 对于0 padding的特殊处理。
%         end
        temp = hex2dec(join_string(char_i));
        bin_char_vec = [bin_char_vec, dec2bin(temp, 4)];
    end
    current_bin_char_= 1;
    for subcarrier_index=1:num_subcarrier
        if flag_use_order_protocol
            temp_vec = protocol_order_decode(bin_char_vec(current_bin_char_: current_bin_char_ + per_subcarrier_bits - 1), bphi, bpsi);
            angle_vec(i, subcarrier_index, :) = temp_vec;
        else % use order of wireshark
            temp_vec = wireshark_order_decode(bin_char_vec(current_bin_char_: current_bin_char_ + per_subcarrier_bits - 1), bphi, bpsi);
            angle_vec(i, subcarrier_index, :) = temp_vec;
        end
        current_bin_char_ = current_bin_char_ + per_subcarrier_bits;
    end
end




