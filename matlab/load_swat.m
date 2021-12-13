function data = load_swat(filename)

raw = readtable(filename);
t = datetime(raw.Timestamp,'InputFormat','dd/MM/yyyy hh:mm:ss a');
data = table2timetable(raw(:,2:end), 'RowTimes', t);

end






