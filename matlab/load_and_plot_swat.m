clc
clear all
close all


load('swat_nominal.mat');
[Pn{1},Pn{2},Pn{3},Pn{4},Pn{5},Pn{6}] = parse_swat(swat_nominal);
%plot_swat(Pn{1},Pn{2},Pn{3},Pn{4},Pn{5},Pn{6});

load('swat_attack.mat');
[Pa{1},Pa{2},Pa{3},Pa{4},Pa{5},Pa{6}] = parse_swat(swat_attack);
%plot_swat(Pa{1},Pa{2},Pa{3},Pa{4},Pa{5},Pa{6});

figure;
stackedplot(Pn{2})
title('P2 normal')
figure;
stackedplot(Pa{2})
title('P2 attack')
