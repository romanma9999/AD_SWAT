function  plot_swat_P1(P1)

P1A = get_anomaly_times();
P1A_idx = [1 2 3 21 26 30 33 34 35 36];
nP1A = length(P1A_idx);
P1.Properties.VariableNames{end} = 'Anomaly';
P1names =P1.Properties.VariableNames;

for j = 1:nP1A
  dstart = datetime(P1A(P1A_idx(j)).s,'InputFormat','MM/dd/uuuu HH:mm:ss');
  dend = datetime(P1A(P1A_idx(j)).e,'InputFormat','MM/dd/uuuu HH:mm:ss');
  TR = timerange(dstart,dend);
  P1(TR,:).Anomaly(:) = 1;
end

P1var = P1.Variables;
nP1var = size(P1var,2);
P1time = P1.Time;

figure;

for i = 1:nP1var
subplot(nP1var,1,i)
prev_start = 1;
for j = 1:nP1A
  dstart = datetime(P1A(P1A_idx(j)).s,'InputFormat','MM/dd/uuuu HH:mm:ss');
  dend = datetime(P1A(P1A_idx(j)).e,'InputFormat','MM/dd/uuuu HH:mm:ss');
  istart = find(P1time == dstart)-1;
  iend = find(P1time == dend);
  plot(P1time(prev_start:istart),P1var(prev_start:istart,i),'b');
  hold on 
  plot(P1time(istart:iend),P1var(istart:iend,i),'r');
  hold on 
  prev_start = iend;
end
  plot(P1time(iend:end),P1var(iend:end,i),'b');

legend(P1names(i));
xlabel('datetime');
ylabel(P1names(i));
grid on;
end

figure;
stackedplot(P1)
title('P1')

end