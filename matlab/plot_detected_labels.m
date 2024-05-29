clc
close all
clear all

postfix = 'diff_h1';
%filename = 'swat_htm_results_learn_mixed';
filename = 'swat_htm_results_learn_mixed_diff';
fullpath = ['../HTM_results/' filename];

values = [1:6];

first = 1;
for i = 1:length(values)
    dl_fullname = [fullpath '_dl_P'  num2str(values(i)) '.csv' ];
    tfp_fullname = [fullpath '_TFP_P'  num2str(values(i)) '.csv' ];
    tprms_fullname = [fullpath '_params_P'  num2str(values(i)) '.csv' ];
    final_fullname = [fullpath '_final.csv' ];
    if first 
        data.dlt =readtable(dl_fullname);
        data.tfpt =readtable(tfp_fullname);
        data.tprms = readtable(tprms_fullname);
        data.finalt = readtable(final_fullname);
        first = 0;
    else
        data.tprms = [data.tprms readtable(tprms_fullname)];
        data.dlt = [data.dlt readtable(dl_fullname)];
        data.tfpt = [data.tfpt readtable(tfp_fullname)];
    end
end

save(['../labels_data_' postfix '.mat'],'data')
figure;
xvalues = 1:41;
yvalues =data.dlt.Properties.VariableNames;

val = data.dlt.Variables;
val(4,:) = nan;
val(5,:) = nan;
val(9,:) = nan;
val(12:15,:) = nan;
val(18,:) = nan;
val(29,:) = nan;
h = heatmap(xvalues,yvalues,val','CellLabelColor','none','ColorbarVisible','off');
xlabel('attack index')
ylabel('channel name')

plot_tfp_bar(data.tfpt.Properties.VariableNames,data.tfpt.Variables)
plot_tfp_bar(data.finalt.Properties.VariableNames,data.finalt.Variables)
plot_params(data.tprms.Properties.VariableNames,data.tprms.Variables)
