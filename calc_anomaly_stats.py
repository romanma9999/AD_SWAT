import sys
import swat_utils
import argparse
import pandas as pd
import xlsxwriter
import xlrd

parser = argparse.ArgumentParser(description='runtime configuration for HTM anomaly statistics on SWAT')
parser.add_argument('--stage_name', '-sn', metavar='STAGE_NAME', default='P1', choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'], type=str.upper)
parser.add_argument('--channel_name', '-cn', metavar='CHANNEL_NAME',type=str)
parser.add_argument('--batch_channel_names', '-bcn', metavar='BATCH_CHANNEL_NAMES',default = "",type=str)
parser.add_argument('--batch_file_names', '-bfn', metavar='BATCH_FILE_NAMES',default = "",type=str)
parser.add_argument('--freeze_type', '-ft', default='off', choices=['off', 'during_training', 'end_training'], type=str.lower)
parser.add_argument('--learn_type', '-lt', default='always', choices=['always', 'train_only'], type=str.lower)
parser.add_argument('--prefix', default="", type=str.lower)
parser.add_argument('--output_file_path', default="./HTM_results/", type=str)
parser.add_argument('--raw_threshold', '-rth', default=0.3, type=float, help="raw anomaly score threshold")
parser.add_argument('--logl_threshold', '-lglth', default=0.6, type=float, help="log likelihood anomaly score threshold")
parser.add_argument('--mean_window', '-rw', default=9, type=int, help="moving mean anomaly score window")
parser.add_argument('--mean_threshold', '-mth', default=0.6, type=float, help="moving mean anomaly score threshold")
parser.add_argument('--sum_window', '-sw', default=119, type=int, help="moving sum anomaly score window")
parser.add_argument('--sum_threshold', '-sth', default=0.6, type=float, help="moving sum anomaly score threshold")
parser.add_argument('--training_count', '-tc', default=414000, type=int, help="training points count")
parser.add_argument('--excel_filename', '-excel', default="swat_htm_results", type=str)


def process_score(labels,scores,score_name,threshold,output_path,file_prefix):
  anomaly_score_output_path = ''.join([output_path, file_prefix, f'_{score_name}_anomaly_score.pred'])
  thresholded_scores = swat_utils.anomaly_score(scores, threshold)
  swat_utils.save_list(thresholded_scores, anomaly_score_output_path)
  stats = swat_utils.calc_anomaly_stats(thresholded_scores,labels)
  stats['Threshold'] = threshold
  return stats

def unify_detection_delay_list(l_count,stats):
  detection_delay = [-1] * l_count
  for idx in range(l_count):
    first_update = True
    for s in stats.values():
      label_idx = idx + 1
      detected_labels = s['detected_labels']
      if label_idx in detected_labels:
          delay_idx = detected_labels.index(label_idx)
          delay_value = s['detection_delay'][delay_idx]
          if first_update:
            detection_delay[idx] = delay_value
            first_update = False
          else:
            if delay_value < detection_delay[idx]:
              detection_delay[idx] = delay_value

  return [x for x in detection_delay if x != -1]

def unify_stats(stats):
  TP_detected_labels = set()
  first = True
  fpa = []
  for key, s in stats.items():
    TP_detected_labels |= set(s['detected_labels'])
    if first:
      fpa = s['fp_array']
      first = False
    else:
      fpa = [x|y for x,y in zip(fpa,s['fp_array'])]
    del s['fp_array']

  FP = sum([1 for idx, x in enumerate(fpa[:-1]) if fpa[idx] == 0 and fpa[idx+1] == 1])
  l_count = next(iter(stats.values()))['LabelsCount']
  TP_detection_delay = unify_detection_delay_list(l_count, stats)

  TP = len(TP_detected_labels)
  FN = l_count - TP
  PR = swat_utils.precision(TP, FP)
  RE = swat_utils.recall(TP, FN)


  stats = {}
  stats['TP'] = TP
  stats['FP'] = FP
  stats['FN'] = FN
  stats['PR'] = PR
  stats['RE'] = RE
  stats['F1'] = swat_utils.F1(PR, RE)
  stats['detected_labels'] = list(TP_detected_labels)
  stats['detection_delay'] = TP_detection_delay
  stats['fp_array'] = fpa
  stats['LabelsCount'] = l_count

  return stats


def get_key_with_max_F1(stats):
  max_value = 0
  max_key = ''
  for key, value in stats.items():
    if value['F1'] >= max_value:
      max_value = value['F1']
      max_key = key

  return max_key

def get_channel_stats(args,channel_name,input_prefix,label_prefix):
  print(f'process {channel_name}.')
  input_filepath = ''.join([args.output_file_path, input_prefix, '_res.csv'])
  label_filepath = ''.join([args.output_file_path, label_prefix, '_attack.real'])

  print("read HTM results data..")
  dl = pd.read_csv(label_filepath, header=None)
  labels = dl.iloc[:, 0]

  df = pd.read_csv(input_filepath)
  raw_scores = df.iloc[args.training_count:, 3]
  logl_scores = df.iloc[args.training_count:, 4]
  mean_scores = df.iloc[args.training_count:, 3].rolling(args.mean_window, min_periods=1, center=False).mean()
  sum_scores = df.iloc[args.training_count:, 3].rolling(args.sum_window, min_periods=1, center=False).sum()
  print("process anomaly scores...")
  stats = {}
  stats['raw'] = process_score(labels, raw_scores, 'raw', args.raw_threshold, args.output_file_path, input_prefix)
  stats['logl'] = process_score(labels, logl_scores, 'logl', args.logl_threshold, args.output_file_path, input_prefix)
  stats['mean'] = process_score(labels, mean_scores, 'mean', args.mean_threshold, args.output_file_path, input_prefix)
  stats['sum'] = process_score(labels, sum_scores, 'sum', args.sum_threshold, args.output_file_path, input_prefix)

  max_stats = {}
  max_key = get_key_with_max_F1(stats)
  max_stats[max_key] = stats.pop(max_key)

  print(f'{max_key} score selected, F1 = {max_stats[max_key]["F1"]}')
  max_key = get_key_with_max_F1(stats)
  max_stats[max_key] = stats.pop(max_key)

  print(f'{max_key} score selected, F1 = {max_stats[max_key]["F1"]}')
  max_stats[channel_name] = unify_stats(max_stats)

  return max_stats

def write_stats_to_excel(worksheet, raw, stats,format):
  for idx, (key,value) in enumerate(stats.items()):
    if type(value) is list:
      val = str(value)
      worksheet.write(raw, idx+1, val,format)
      worksheet.set_column(idx+1, idx+1, len(val))
    else:
      worksheet.write(raw,idx+1, value,format)

  return

def save_to_excel(filename, stats):
  workbook = open_file_for_update('swat_htm_results.xlsx')
  worksheet = workbook.add_worksheet(args.stage_name)
  bold = workbook.add_format({'bold': True})
  format_head = workbook.add_format({'align': 'center'})
  format_head.set_bold()
  format_cells = workbook.add_format({'align': 'center'})

  first = True
  idx_channel = 1
  for _,(_,s) in enumerate(stats.items()):
    for (channel_name, channel) in s.items():
      worksheet.write(idx_channel, 0, channel_name,bold)
      write_stats_to_excel(worksheet, idx_channel, channel, format_cells)
      idx_channel += 1
      # headings
      if first:
        first = False
        worksheet.set_column(0, 0, 8)
        for idx, key in enumerate(channel):
            worksheet.write(0, idx + 1, key,format_head)
            worksheet.set_column(idx+1, idx+1, max(8, len(key)+1))

  workbook.close()

  return


def open_file_for_update(filename):
  workbook = xlsxwriter.Workbook(filename)
  wbRD  = xlrd.open_workbook(filename)
  sheets = wbRD.sheets()
  # run through the sheets and store sheets in workbook
  # this still doesn't write to the file yet
  for sheet in sheets: # write data from old file
      newSheet = workbook.add_worksheet(sheet.name)
      for row in range(sheet.nrows):
          for col in range(sheet.ncols):
              newSheet.write(row, col, sheet.cell(row, col).value)

  return workbook

def main(args):

  if len(args.batch_channel_names) != 0:
    channels_list = [x for x in args.batch_channel_names.split(',')]
    channels_filenames_list = [x for x in args.batch_file_names.split(',')]
  else:
    channels_list = [args.channel_name]
    channels_filenames_list = [swat_utils.get_file_prefix(args, args.channel_name)]

  first = True
  stats = {}
  channel_stats = {}
  for channel_name,file_prefix in zip(channels_list,channels_filenames_list):
    args.channel_name = channel_name
    if first:
      label_prefix = swat_utils.get_file_prefix(args, args.channel_name)
      first = False
    stats[channel_name] = get_channel_stats(args,channel_name,file_prefix,label_prefix)
    channel_stats[channel_name] = stats[channel_name][channel_name]

  stats[args.stage_name] = {args.stage_name: unify_stats(channel_stats)}
  del stats[args.stage_name][args.stage_name]['fp_array']

  for _, s in stats.items():
    for key, value in s.items():
      print(f'{key:5}:{value}')

  save_to_excel(args.excel_filename,stats)

  return

def test_unify_detection_delay_list():
  l_count = 6
  s1 = {}
  s1['detected_labels'] = [1, 2, 5]
  s1['detection_delay'] = [10,20,30]

  s = {}
  s['1'] = s1

  delay_list = unify_detection_delay_list(l_count,s)
  print(f'2: {delay_list}')

  s1 = {}
  s1['detected_labels'] = [1, 2, 5]
  s1['detection_delay'] = [10, 20, 30]
  s2 = {}
  s2['detected_labels'] = [2 ,3 ,5]
  s2['detection_delay'] = [1, 2, 40]

  s = {}
  s['1'] = s1
  s['2'] = s2
  delay_list = unify_detection_delay_list(l_count,s)
  print(f'2: {delay_list}')

if __name__ == "__main__":

    # test_unify_detection_delay_list()

    sys.argv = ['calc_anomaly_stats.py',
                '--stage_name', 'P1',
                '-bcn', 'LIT101,P102',
                '-bfn','P1_LIT101_learn_always_freeze_off,P1_P102_learn_train_only_freeze_off',
                '--freeze_type', 'off',
                '--learn_type', 'always',
                '--raw_threshold', '0.7']




    args = parser.parse_args()
    print(args)
    main(args)