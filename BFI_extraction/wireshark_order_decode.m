function returnvec = wireshark_order_decode(tar_string, bphi, bpsi)
    current_bin_char = 1;
    phi11 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi21 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi31 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi22 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    phi32 = bin2dec(tar_string(current_bin_char:current_bin_char+bphi-1)); current_bin_char = current_bin_char + bphi;
    psi21 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi31 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi41 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi32 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    psi42 = bin2dec(tar_string(current_bin_char:current_bin_char+bpsi-1)); current_bin_char = current_bin_char + bpsi;
    returnvec =  [phi11, phi21, phi31, psi21, psi31, psi41, phi22, phi32, psi32, psi42];
end