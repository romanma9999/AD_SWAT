function  P_plot(P,var_to_plot,PID)

[PA,PA_idx] = get_anomaly_times(PID);
nPA = length(PA_idx);
Pnames =P.Properties.VariableNames;

    
Pvar = P.Variables;
n_var_to_plot = length(var_to_plot);
if n_var_to_plot == 0
     n_var_to_plot = size(Pvar,2);
     var_to_plot = 1:n_var_to_plot;
end

%Pvar = (Pvar - min(Pvar))./(max(Pvar) - min(Pvar));
Ptime = P.Time;

h = figure;
tiledlayout(n_var_to_plot,1)
ax = zeros(1,n_var_to_plot);
for i = 1:n_var_to_plot
%subplot(n_var_to_plot,1,i)
t = nexttile;
ax(i) = t;
prev_start = 1;
for j = 1:nPA
  dstart = datetime(PA(PA_idx(j)).s,'InputFormat','MM/dd/uuuu HH:mm:ss');
  dend = datetime(PA(PA_idx(j)).e,'InputFormat','MM/dd/uuuu HH:mm:ss');
  istart = find(Ptime == dstart)-1;
  iend = find(Ptime == dend);
  plot(Ptime(prev_start:istart),Pvar(prev_start:istart,var_to_plot(i)),'b');
  hold on 
  plot(Ptime(istart:iend),Pvar(istart:iend,var_to_plot(i)),'r','LineWidth',2);
  hold on 
  prev_start = iend;
end
  plot(Ptime(iend:end),Pvar(iend:end,var_to_plot(i)),'b');

legend(Pnames(i),'Interpreter', 'none');
xlabel('datetime');
ylabel(Pnames(i),'Interpreter', 'none');
grid on;
end

linkaxes(ax,'x');
ZoomHandle = zoom(h);
set(ZoomHandle,'Motion','horizontal');

% figure;
% stackedplot(P)
% title('P')

end