clc
clear all
close all
%swat_nominal = load_swat('SWaT_Dataset_Normal_v1.xlsx');
%swat_attack = load_swat('SWaT_Dataset_Attack_v0.xlsx');
%save('swat_nominal.mat','swat_nominal');
%save('swat_attack.mat','swat_attack');

%  load('swat_nominal.mat');
%  [P1n,P2n,P3n,P4n,P5n,P6n] = parse_swat(swat_nominal);
%  plot_swat(P1n,P2n,P3n,P4n,P5n,P6n);

load('swat_attack.mat');
[P1a,P2a,P3a,P4a,P5a,P6a] = parse_swat(swat_attack);
plot_swat(P1a,P2a,P3a,P4a,P5a,P6a);

% deprecate features MADICS
% variance is 0 :  P202, P301, P401, P404, P502, P601
% K-S Statistic higher than 0.25 : AIT402, AIT201, AIT501, AIT502, AIT202, AIT504, FIT301, PIT502, PIT503, FIT503, PIT501, AIT203, AIT401, AIT503, FIT601
% Training and test empirical distributions do not match : P201
%
%
%
%
