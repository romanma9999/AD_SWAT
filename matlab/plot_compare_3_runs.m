function plot_compare_3_runs()
clc
close all
clear all

is_abs = true;
is_diff = false;

if is_abs
    name1 = 'abs_h1';
    name2 = 'abs_h2';
    name3 = 'abs_h3';
    filter_enabled = [true ,true, true];
end
if is_diff
    name1 = 'diff_h1';
    name2 = 'diff_h2';
    name3 = 'diff_h3';
end

d1 = load(['../labels_data_' name1 '.mat']);
d2 = load(['../labels_data_' name2 '.mat']);
d3 = load(['../labels_data_' name3 '.mat']);


show_diff_only = false;


figure;
xvalues = 1:41;
yvalues =d1.data.dlt.Properties.VariableNames;
val1 = extract_val(d1.data,1);
val2 = extract_val(d2.data,2);
val3 = extract_val(d3.data,4);

if is_abs
    if filter_enabled(1)
        val1(:,4) = []; 
        yvalues(:,4) = []; 
        val1(:,32) = []; 
        yvalues(:,32) = [];   
    end
    if filter_enabled(2)
        val2(:,28) = []; 
    end
    if filter_enabled(3)
        val3(:,28) = []; 
    end
else
      val1(:,4) = []; 
      yvalues(:,4) = []; 
end

val = val1 + val2 + val3;
if show_diff_only
    val(val == 3) = 0;
    val(val == 5) = 0;
    val(val == 6) = 0;
    val(val == 7) = 0;
end
%h = heatmap(xvalues,yvalues,val','CellLabelColor','none','ColorbarVisible','off');
%h = heatmap(xvalues,yvalues,val','ColorbarVisible','off');
h = heatmap(xvalues,yvalues,val');
xlabel('attack index')
ylabel('channel name')
title('L1 detection matrix for different TSSE input SDR sequence length')
axs = struct(gca); %ignore warning that this should be avoided
cb = axs.Colorbar;
cb.Ticks = [0,1,2,3,4,5,6,7];
%cb.TickLabels = {'undetected','auto','with tuning','all detected'};
cb.TickLabels = {'none','1 SDR','2 SDRs','1 or 2 SDRs','3 SDRs','1 or 3 SDRs','2 or 3 SDRs','All'};
h.MissingDataLabel = 'invalid attack';
colormap([1 1 1; 
                  0.5 0.6 0.7 ; 
                  0.4 0.4 0.7; 
                  0.1 0.8 0.1; 
                  0.8 0.1 0.1; 
                  0.1 0.1 0.8;
                  0.8 0.1 0.8; 
                  0.1 0.8 0.8]);

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
