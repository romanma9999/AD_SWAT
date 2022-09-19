function  plot_tfp_bar(names, values)

figure;
channel_names =categorical(names);
tp_values = values(1,:);
fp_values = values(2,:);

w1 = 0.5; 
w2 = .25;

x=1:length(tp_values);
bar(x,tp_values,w1,'FaceColor',[0.2 0.2 0.5])
xticks(x)
xticklabels(channel_names)
hold on
bar(x,fp_values,w2,'FaceColor',[0 0.7 0.7])
hold off
grid on
xlabel('channel name')
ylabel('detections count')
legend({'True Positive','False Positive'},'Location','northwest')

end

