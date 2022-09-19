clc
close all
clear all

filename = '../swat_htm_results_learn_mixed';
%filename = '../swat_htm_results_learn_mixed_diff';
values = [1:6];

first = 1;
for i = 1:length(values)
    dl_fullname = [filename '_dl_P'  num2str(values(i)) '.csv' ];
    tfp_fullname = [filename '_TFP_P'  num2str(values(i)) '.csv' ];
    final_fullname = [filename '_final.csv' ];
    if first 
        dlt =readtable(dl_fullname);
        tfpt =readtable(tfp_fullname);
        finalt = readtable(final_fullname);
        first = 0;
    else
        dlt = [dlt readtable(dl_fullname)];
        tfpt = [tfpt readtable(tfp_fullname)];
    end
end

figure;
xvalues = 1:41;
yvalues =dlt.Properties.VariableNames;
val = dlt.Variables;
val(4,:) = nan;
val(5,:) = nan;
val(9,:) = nan;
val(12:15,:) = nan;
val(18,:) = nan;
val(29,:) = nan;
h = heatmap(xvalues,yvalues,val');
xlabel('attack index')
ylabel('channel name')

plot_tfp_bar(tfpt.Properties.VariableNames,tfpt.Variables)
plot_tfp_bar(finalt.Properties.VariableNames,finalt.Variables)

