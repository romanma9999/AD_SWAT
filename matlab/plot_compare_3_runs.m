function plot_compare_3_runs()
clc
close all
clear all

name1 = 'abs1';
name2 = 'abs10';
name3 = 'abs20';

d1 = load(['../labels_data_' name1 '.mat']);
d2 = load(['../labels_data_' name2 '.mat']);
d3 = load(['../labels_data_' name3 '.mat']);

figure;
xvalues = 1:41;
yvalues =d1.data.dlt.Properties.VariableNames;
val1 = extract_val(d1.data,1);
val2 = extract_val(d2.data,2);
val3 = extract_val(d3.data,4);
val = val1 + val2 + val3;
val(val == 3) = 0;
val(val == 5) = 0;
val(val == 6) = 0;
val(val == 7) = 0;
%h = heatmap(xvalues,yvalues,val','CellLabelColor','none','ColorbarVisible','off');
%h = heatmap(xvalues,yvalues,val','ColorbarVisible','off');
h = heatmap(xvalues,yvalues,val');
xlabel('attack index')
ylabel('channel name')
axs = struct(gca); %ignore warning that this should be avoided
cb = axs.Colorbar;
cb.Ticks = [0,1,2,3];
%cb.TickLabels = {'undetected','auto','with tuning','all detected'};
cb.TickLabels = {'undetected','Ts=1s','Ts=10s','Ts=20s'};
h.MissingDataLabel = 'invalid attack';
colormap([1 1 1; 0.5 0.6 0.7; 0.4 0.4 0.7; 0.1 0.8 0.1]);

function val = extract_val(data,multiplier)

val = data.dlt.Variables;
mm = multiplier*ones(size(val));
val = val.*mm;
val(4,:) = nan;
val(5,:) = nan;
val(9,:) = nan;
val(12:15,:) = nan;
val(18,:) = nan;
val(29,:) = nan;
