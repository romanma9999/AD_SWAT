function  P_plot(P,plot_cfg,AnomalyIdx,PID)

[PA,PA_idx] = get_anomaly_times(PID);
nPA = length(PA_idx);
Pnames =P.Properties.VariableNames;
   
Pvar = P.Variables;

if isempty(plot_cfg) 
     n_var_to_plot = size(Pvar,2);
     var_to_plot = 1:n_var_to_plot;
else    
    var_to_plot = plot_cfg(1,:);
    n_var_to_plot = length(var_to_plot);
end

Pvar = (Pvar - min(Pvar))./(max(Pvar) - min(Pvar));
Ptime = P.Time;

figure;
for i = 1:n_var_to_plot
    idx = var_to_plot(i);
    addon = (n_var_to_plot - i)*2;
    if idx == AnomalyIdx 
      plot(Ptime,addon + Pvar(:,idx),'r','LineWidth',1,'DisplayName','Attack');
      hold on;
    else
      plot(Ptime,addon + Pvar(:,idx),'DisplayName',Pnames{idx});
      hold on;
    end
    xlabel('datetime');
    grid on;
end
legend show

end