%% 数据读取部分
fname = 'try_prove_stable_bfi.json'; 
% fname = '5.json'; 
[epoch_time_seq, angle_mat] = decode_bfi(fname);

subcarrier_idx = 25;
subcarrier_idx_2 = 30;
phi_seq = struct('PHI11', [],...
                             'PHI21', [],...
                             'PHI31', [],...
                             'PHI22', [],...
                             'PHI32', []...
                            );
psi_seq =struct('PSI21', [],...
                            'PSI31', [],...
                            'PSI41', [],...
                            'PSI32', [],...
                            'PSI42', []...
                            );
%%
frame_length = length(epoch_time_seq);
% returnvec =  [phi11, phi21, phi31, psi21, psi31, psi41, phi22, phi32, psi32, psi42];
for idx = 1:frame_length
    phi_seq.PHI11 = [phi_seq.PHI11, angle_mat(idx, subcarrier_idx, 1)];
    phi_seq.PHI21 = [phi_seq.PHI21, angle_mat(idx, subcarrier_idx, 2)];
    phi_seq.PHI31 = [phi_seq.PHI31, angle_mat(idx, subcarrier_idx, 3)];
    psi_seq.PSI21 = [psi_seq.PSI21, angle_mat(idx, subcarrier_idx, 4)];
    psi_seq.PSI31 = [psi_seq.PSI31, angle_mat(idx, subcarrier_idx, 5)];
    psi_seq.PSI41 = [psi_seq.PSI41, angle_mat(idx, subcarrier_idx, 6)];
    phi_seq.PHI22 = [phi_seq.PHI22, angle_mat(idx, subcarrier_idx, 7)];
    phi_seq.PHI32 = [phi_seq.PHI32, angle_mat(idx, subcarrier_idx, 8)];
    psi_seq.PSI32 = [psi_seq.PSI32, angle_mat(idx, subcarrier_idx, 9)];
    psi_seq.PSI42 = [psi_seq.PSI42, angle_mat(idx, subcarrier_idx, 10)];
end
angle_index_mat = [phi_seq.PHI11.', phi_seq.PHI21.', phi_seq.PHI31.', psi_seq.PSI21.', psi_seq.PSI31.', psi_seq.PSI41.'];
angle_index_mat  = [ angle_index_mat , phi_seq.PHI22.', phi_seq.PHI32.', psi_seq.PSI32.', psi_seq.PSI42.'];
%%
% returnvec =  [phi11, phi21, phi31, psi21, psi31, psi41, phi22, phi32, psi32, psi42];
phi_seq = struct('PHI11', [],...
                             'PHI21', [],...
                             'PHI31', [],...
                             'PHI22', [],...
                             'PHI32', []...
                            );
psi_seq =struct('PSI21', [],...
                            'PSI31', [],...
                            'PSI41', [],...
                            'PSI32', [],...
                            'PSI42', []...
                            );
for idx = 1:frame_length
    phi_seq.PHI11 = [phi_seq.PHI11, angle_mat(idx, subcarrier_idx_2, 1)];
    phi_seq.PHI21 = [phi_seq.PHI21, angle_mat(idx, subcarrier_idx_2, 2)];
    phi_seq.PHI31 = [phi_seq.PHI31, angle_mat(idx, subcarrier_idx_2, 3)];
    psi_seq.PSI21 = [psi_seq.PSI21, angle_mat(idx, subcarrier_idx_2, 4)];
    psi_seq.PSI31 = [psi_seq.PSI31, angle_mat(idx, subcarrier_idx_2, 5)];
    psi_seq.PSI41 = [psi_seq.PSI41, angle_mat(idx, subcarrier_idx_2, 6)];
    phi_seq.PHI22 = [phi_seq.PHI22, angle_mat(idx, subcarrier_idx_2, 7)];
    phi_seq.PHI32 = [phi_seq.PHI32, angle_mat(idx, subcarrier_idx_2, 8)];
    psi_seq.PSI32 = [psi_seq.PSI32, angle_mat(idx, subcarrier_idx_2, 9)];
    psi_seq.PSI42 = [psi_seq.PSI42, angle_mat(idx, subcarrier_idx_2, 10)];
end
angle_index_mat_2 = [phi_seq.PHI11.', phi_seq.PHI21.', phi_seq.PHI31.', psi_seq.PSI21.', psi_seq.PSI31.', psi_seq.PSI41.'];
angle_index_mat_2 = [ angle_index_mat_2 , phi_seq.PHI22.', phi_seq.PHI32.', psi_seq.PSI32.', psi_seq.PSI42.'];

%% 数据处理部分
time_sequence = (epoch_time_seq - epoch_time_seq(1));
num_frame = length(time_sequence);
new_phi_seq = phi_seq;
field_name_phi = fieldnames(phi_seq);
for id_field = 1:numel(field_name_phi)
    field_name = field_name_phi{id_field};
    tar_seq = phi_seq.(field_name);
    tar_seq = ((tar_seq/64) * 2*pi)/pi * 180;
    new_phi_seq.(field_name) = tar_seq;
end

new_psi_seq = psi_seq;
field_name_psi = fieldnames(psi_seq);
for id_field = 1:numel(field_name_psi)
    field_name = field_name_psi{id_field};
    tar_seq = psi_seq.(field_name);
    tar_seq = ((tar_seq/16) * 2*pi)/pi * 180;
    new_psi_seq.(field_name) = tar_seq;
end


Nr = 4; Nc = 2;
NumBitsPhi = 6; NumBitsPsi=4;

%% 画图部分

figure(10);
plot_y = zeros(1,size(angle_index_mat,1));
for idx_time = 1:size(angle_index_mat,1)
    
%     plot_y(idx_time) = mean(angle(V(idx_time, :, :)./repmat(V(idx_time,:,end),[1,1,4])), 'all');
    V = bfDecompress((angle_index_mat(idx_time,:)),Nr,Nc,NumBitsPhi,NumBitsPsi);
    V_2 = bfDecompress([angle_index_mat_2(idx_time,:)],Nr,Nc,NumBitsPhi,NumBitsPsi);
    plot_y(idx_time) = sum(abs(V(1, 1, :)).^2);
    plot_y(idx_time) = abs(mean(V(1,1,:)));
    plot_y(idx_time) = abs(sum(V(1,1,:) .* conj(V(1,2,:))));
    plot_y(idx_time) = mean(angle(V(1,1,1)));
end
hold on;
scatter(time_sequence, plot_y, 'Marker', '.');
scatter(time_sequence, plot_y+pi/2, 'Marker', '.');
scatter(time_sequence, plot_y+pi, 'Marker', '.');
scatter(time_sequence, plot_y+pi*3/2, 'Marker', '.');
% return
% PHI figures
smooth_level = 1;
figure(1);
field_name_phi = fieldnames(new_phi_seq);
num_fields = numel(field_name_phi);
for idx_field = 1:num_fields
    field_name = field_name_phi{idx_field};
    subplot(num_fields,1,idx_field);
    scatter(time_sequence, smooth(new_phi_seq.(field_name), smooth_level),'Marker','.');
    xlabel('Time [Sec]'); ylabel('Angle [Degree]'); title(['PHI ',field_name]); xlim([0,45]);
end

figure(2);
field_name_phi = fieldnames(new_psi_seq);
num_fields = numel(field_name_psi);
for idx_field = 1:num_fields
    field_name = field_name_psi{idx_field};
    subplot(num_fields,1,idx_field);
    scatter(time_sequence, smooth(new_psi_seq.(field_name), smooth_level),'Marker','.');
    xlabel('Time [Sec]'); ylabel('Angle [Degree]'); title(['PSI ', field_name]); xlim([0,45]);
end




