function plot_compare_2_runs()
clc
close all
clear all

graphn = 6;

if graphn == 1
    name1 = 'diff1';
    name2 = 'diff1_tuning';
end
if graphn == 2
    name1 = 'abs1';
    name2 = 'diff1';
end
if graphn == 3
    name1 = 'abs1_madics';
    name2 = 'diff1_madics';
end
if graphn == 4
    name1 = 'abs1_delta_madics';
    name2 = 'diff1_delta_madics';
end
if graphn == 5
    name1 = 'abs1';
    name2 = 'diff1';
end
if graphn == 6
    name1 = 'abs_h3';
    name2 = 'diff_h2';
end

d1 = load(['../labels_data_' name1 '.mat']);
d2 = load(['../labels_data_' name2 '.mat']);

figure;
xvalues = 1:41;
yvalues =d1.data.dlt.Properties.VariableNames;
val1 = extract_val(d1.data,1);
val2 = extract_val(d2.data,2);
if graphn == 2
    val2(:,29) = []; % filter out fit502 channel
end
if graphn == 3
    val2(:,21) = []; % filter out fit502 channel
end
if graphn == 5
    val2(:,29) = []; % filter out fit502 channel
end
if graphn == 6
    val1(:,28) = []; % filter out fit502 channel
    val2(:,28) = []; % filter out fit502 channel
    yvalues(:,28) = []; % filter out fit502 channel    
end


size(val1)
size(val2)

if graphn == 5 
    val = val1.*val2;
else
    val = val1 + val2;
end

val(4,:) = nan;
val(5,:) = nan;
val(9,:) = nan;
val(12:15,:) = nan;
val(18,:) = nan;
val(29,:) = nan;

%h = heatmap(xvalues,yvalues,val','CellLabelColor','none','ColorbarVisible','off');
%h = heatmap(xvalues,yvalues,val','ColorbarVisible','off');
h = heatmap(xvalues,yvalues,val');
title('L1 detection matrix for feature channels dAk and Ak+TSSE')
xlabel('attack index')
ylabel('channel name')
axs = struct(gca); %ignore warning that this should be avoided
cb = axs.Colorbar;
cb.Ticks = [0,1,2,3];
%cb.TickLabels = {'undetected','auto','oracle','all detected'};
%cb.TickLabels = {'undetected','abs h1','abs h2','all detected'};
cb.TickLabels = {'undetected','Ak + TSSE','dAk','All'};
h.MissingDataLabel = 'invalid attack';
colormap([1 1 1; 0.9 0.2 0.3; 0.9 0.6 0.2; 0.1 0.8 0.1]);

function val = extract_val(data,multiplier)

val = data.dlt.Variables;
mm = multiplier*ones(size(val));
val = val.*mm;




