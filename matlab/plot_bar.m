function plot_bar(values, x_tick_names, x_axis_name, y_axis_name)

figure
w1 = 0.5; 
x=1:length(values);
bar(x,values,w1,'FaceColor',[0.2 0.2 0.5])
xticks(x)
xticklabels(x_tick_names)
grid on
xlabel(x_axis_name)
ylabel(y_axis_name)
legend(y_axis_name,'Location','northwest')

end

