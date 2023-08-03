function Vtilda = bfDecompress(angidx,Nr,Nc,bphi,bpsi)
%bfDecompress re-constructs Beamforming feedback matrix
%   V = bfDecompress(ANGIDX,NR,NC,BPSI,BPHI) reconstructs the
%   beamforming feedback matrix, V from ANGIDX. ANGIDX are the quantized
%   angles. NR and NC are the number of rows and number of columns in V.
%   There are two kinds of angles in ANGIDX: phi and psi. Those angles are
%   quantized according to the bit resolution given by BPHI and BPSI for
%   phi and psi respectively. The size of ANGIDX should be of the form
%   (Number of active sub-carriers)X(Number of angles for a sub-carrier).
%
%   References:
%   1) IEEE Standard for Information technology--Telecommunications and
%   information exchange between systems Local and metropolitan area
%   networks--Specific requirements - Part 11: Wireless LAN Medium Access
%   Control (MAC) and Physical Layer (PHY) Specifications," in IEEE Std
%   802.11-2016 (Revision of IEEE Std 802.11-2012) , vol., no., pp.1-3534,
%   Dec. 7 2016.

%   Copyright 2018 The MathWorks, Inc.

p = min([Nc,Nr-1]);
[Nst,NumAngles] = size(angidx);

% Perform dequantization first. See table 9-68 (Quantization of angles) in [1]
angles = zeros(NumAngles,1,Nst);
angcnt = 1;
for ii = Nr-1:-1:max(Nr-Nc,1)
    for jj = 1:ii
        angles(angcnt,1,:) = (2*angidx(:,angcnt)+1)*pi/(2^bphi);
        angcnt = angcnt + 1;
    end
    
    for jj = 1:ii
        angles(angcnt,1,:) = (2*angidx(:,angcnt)+1)*pi/(2^(bpsi+2));
        angcnt = angcnt + 1;
    end
end

% Construction of V matrix from the angles.
V = repmat(eye(Nr,Nc),[1,1,Nst]);
NumAnglesCnt = NumAngles;
for ii = p:-1:1 % Eq 19-85 in [1].
    for jj = Nr:-1:ii+1
        % for each jj, construct Givens matrix, G
        for sc = 1:Nst
            Gt = eye(Nr); % G transpose
            Gt(ii,ii) = cos(angles(NumAnglesCnt,1,sc));
            Gt(ii,jj) = -1*sin(angles(NumAnglesCnt,1,sc));
            Gt(jj,ii) = sin(angles(NumAnglesCnt,1,sc));
            Gt(jj,jj) = cos(angles(NumAnglesCnt,1,sc));
            V(:,:,sc) = Gt*V(:,:,sc);
        end
        NumAnglesCnt = NumAnglesCnt - 1;
    end
    D = [ones(ii-1,1,Nst); exp(1j*angles(NumAnglesCnt-Nr+ii+1:NumAnglesCnt,1,:)); ones(1,1,Nst)];
    NumAnglesCnt = NumAnglesCnt - Nr + ii;
    V = D.*V;
end
Vtilda = permute(V,[3 2 1]);
end