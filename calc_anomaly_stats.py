import sys
import swat_utils
import argparse
import pandas as pd
import xlsxwriter
import xlrd
import os
from os.path import exists
import copy
import swat_htm
import json

parser = argparse.ArgumentParser(description='runtime configuration for HTM anomaly statistics on SWAT')
parser.add_argument('--stage_name', '-sn', metavar='STAGE_NAME', default='P1', choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'], type=str.upper)
parser.add_argument('--channel_name', '-cn', metavar='CHANNEL_NAME',type=str)
parser.add_argument('--batch_channel_names', '-bcn', metavar='BATCH_CHANNEL_NAMES',default = "",type=str)
parser.add_argument('--freeze_type', '-ft', default='off', choices=['off', 'during_training', 'end_training'], type=str.lower)
parser.add_argument('--learn_type', '-lt', default='always', choices=['always', 'train_only'], type=str.lower)
parser.add_argument('--prefix', default="", type=str.lower)
parser.add_argument('--output_file_path', default="./HTM_results/", type=str)
parser.add_argument('--raw_threshold', '-rth', default=0.4, type=float, help="raw anomaly score threshold")
parser.add_argument('--logl_threshold', '-lglth', default=0.6, type=float, help="log likelihood anomaly score threshold")
parser.add_argument('--mean_window', '-rw', default=9, type=int, help="moving mean anomaly score window")
parser.add_argument('--mean_threshold', '-mth', default=0.6, type=float, help="moving mean anomaly score threshold")
parser.add_argument('--grace_time', '-gt', default=1200, type=int, help="unify grace time")
parser.add_argument('--sum_window', '-sw', default=120, type=int, help="moving sum anomaly score window")
parser.add_argument('--sum_threshold', '-sth', default=0.7, type=float, help="moving sum anomaly score threshold")
# parser.add_argument('--training_count', '-tc', default=414000, type=int, help="training points count")
parser.add_argument('--excel_filename', '-efn', default="swat_htm_results", type=str)
parser.add_argument('--excel_sheet_name', '-esn', default="P1", type=str)
parser.add_argument('--output_filename_addon', '-ofa', default="", type=str)
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--training_score', default=False, action='store_true')
parser.add_argument('--channel_type', '-ctype', metavar='CHANNEL_TYPE', default=0, type=int,help='set type 0 for analog, 1 for discrete')
parser.add_argument('--sdr_size', '-size', metavar='SDR_SIZE', default=2048, type=int)
parser.add_argument('--final_stage', default=False, action='store_true')


def process_score(labels,scores,score_name,threshold,output_path,file_prefix,grace_time):
  anomaly_score_output_path = ''.join([output_path, file_prefix, f'_{score_name}_anomaly_score.pred'])
  thresholded_scores = swat_utils.anomaly_score(scores, threshold)
  swat_utils.save_list(thresholded_scores, anomaly_score_output_path)
  stats = swat_utils.calc_anomaly_stats(thresholded_scores,labels,grace_time)
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

def unify_stats(stats,grace_time):
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

  N = len(fpa)
  FP = 0
  FP_arr = [0] * N
  s_now = False
  # FP_start_idx = 0
  for idx, score in enumerate(fpa):
    s_prev = s_now
    s_now = True if score == 1 else False
    if s_now and s_prev == False:
      FP_start_idx = idx

    s_marked = False
    if (s_prev and s_now == False) or (s_now and idx == N-1):
      max_hist = min(FP_start_idx, grace_time)
      for i in range(max_hist):
        if FP_arr[FP_start_idx - i] == 1:
          s_marked = True
          break

      if s_marked == False:
        FP += 1
        FP_arr[FP_start_idx:idx] = [1]*(idx-FP_start_idx)
        if s_now and idx == N-1:
          FP_arr[-1] = 1

  l_count = next(iter(stats.values()))['LabelsCount']
  # TP_detection_delay = unify_detection_delay_list(l_count, stats)
  TP_detection_delay = []

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
  stats['fp_array'] = FP_arr
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

def add_params_info(stats,params):
  for key in stats.keys():
    stats[key]['sampling_interval'] = params['runtime_config']['sampling_interval']
    stats[key]['window'] = params['runtime_config']['window']
    stats[key]['replay_buffer'] = params['runtime_config']['replay_buffer']
    stats[key]['diff_enabled'] = params['runtime_config']['diff_enabled']
    stats[key]['training_only'] = params['runtime_config']['learn_during_training_only']
    stats[key]['freeze'] = params['runtime_config']['freeze_configuration']
    stats[key]['encoding_duration_value'] = params['runtime_config']['encoding_duration_value']
    if 'delay_bins' in params['runtime_config'].keys():
      stats[key]['delay_bins'] = params['runtime_config']['delay_bins']
    else:
      params['runtime_config']['delay_bins'] = [0]

    stats[key]['CustomMinMax'] = params['runtime_config']['CustomMinMax']
    stats[key]['sdr_size'] = params['enc']['size']
    stats[key]['sparsity'] = params['enc']['sparsity']
    stats[key]['sparsity'] = params['enc']['sparsity']


def get_channel_stats(args,channel_name,input_prefix,label_prefix):
  print(f'process {channel_name}.')
  input_filepath = ''.join([args.output_file_path, input_prefix, '_res.csv'])
  label_filepath = ''.join([args.output_file_path, label_prefix, '_attack.real'])
  prm_filepath = ''.join([args.output_file_path, input_prefix, '_param.json'])

  print("read HTM results data..")
  dl = pd.read_csv(label_filepath, header=None)
  df = pd.read_csv(input_filepath)
  with open(prm_filepath) as f:
    params = json.loads(f.read())

  training_count = df.shape[0] - dl.shape[0]
  # print(f'calc training {training_count}, argument {args.training_count}')

  n_grace = int(0.1 * training_count)
  if args.training_score:
    labels = [0]*(training_count - n_grace)
  else:
    labels = dl.iloc[:, 0]

  df = pd.read_csv(input_filepath)
  if args.training_score:
    raw_scores = df.iloc[n_grace:training_count, 3]
    sum_scores = df.iloc[n_grace:training_count, 3].rolling(args.sum_window, min_periods=1, center=False).sum()
    # logl_scores = df.iloc[n_grace:training_count, 4]
    # mean_scores = df.iloc[n_grace:training_count, 3].rolling(args.mean_window, min_periods=1, center=False).mean()
  else:
    raw_scores = df.iloc[training_count:, 3]
    #df['raw_4sum'] = df.iloc[:, 3]
    #df.loc[df['raw_4sum'] >= args.raw_threshold, 'raw_4sum'] = 1.0
    #sum_scores = df.loc[training_count:, 'raw_4sum'].rolling(args.sum_window, min_periods=1, center=False).sum()
    sum_scores = df.iloc[training_count:, 3].rolling(args.sum_window, min_periods=1, center=False).sum()
    # logl_scores = df.iloc[training_count:, 4]
    # mean_scores = df.iloc[training_count:, 3].rolling(args.mean_window, min_periods=1, center=False).mean()


  print("process anomaly scores...")
  stats = {}
  stats['raw'] = process_score(labels, raw_scores, 'raw', args.raw_threshold, args.output_file_path, input_prefix,args.grace_time)
  stats['sum'] = process_score(labels, sum_scores, 'sum', args.sum_threshold, args.output_file_path, input_prefix,args.grace_time)
  # stats['logl'] = process_score(labels, logl_scores, 'logl', args.logl_threshold, args.output_file_path, input_prefix,args.grace_time)
  # stats['mean'] = process_score(labels, mean_scores, 'mean', args.mean_threshold, args.output_file_path, input_prefix,args.grace_time)
  # stats[channel_name] = stats[get_key_with_max_F1(stats)]
  stats[channel_name] = stats['sum']

  add_params_info(stats,params)

  # max_stats = {}
  # max_key = get_key_with_max_F1(stats)
  # max_stats[max_key] = stats.pop(max_key)
  #
  # print(f'{max_key} score selected, F1 = {max_stats[max_key]["F1"]}')
  # max_key = get_key_with_max_F1(stats)
  # max_stats[max_key] = stats.pop(max_key)
  #
  # print(f'{max_key} score selected, F1 = {max_stats[max_key]["F1"]}')
  # max_stats[channel_name] = unify_stats(max_stats,args.grace_time)
  #
  # return max_stats
  return stats

def write_stats_to_excel(worksheet, raw, stats,format):
  for idx, (key,value) in enumerate(stats.items()):
    if type(value) is list:
      val = str(value)
      worksheet.write(raw, idx+1, val,format)
      worksheet.set_column(idx+1, idx+1, len(val))
    else:
      worksheet.write(raw,idx+1, value,format)

  return





def save_to_csv(filename, stats):

  tp_data = {}
  fp_data = {}
  params_data = {}
  LABELS_NUM = 41
  for key, s in stats.items():
    fp_data[key] = [s[key]['TP'], s[key]['FP']]
    params_data[key] = [s[key]['window'], s[key]['sdr_size']]
    tmp = [0]*LABELS_NUM
    labels = s[key]['detected_labels']
    for idx in labels:
      tmp[idx-1] = 1

    tp_data[key] = tmp

  tp_filename = f'{args.output_file_path}{filename}{args.output_filename_addon}_dl_{args.stage_name}.csv'
  tp = pd.DataFrame(tp_data)
  tp.to_csv(tp_filename, sep=',', index=False)
  fp_filename = f'{args.output_file_path}{filename}{args.output_filename_addon}_TFP_{args.stage_name}.csv'
  fp = pd.DataFrame(fp_data)
  fp.to_csv(fp_filename, sep=',', index=False)
  params_filename = f'{args.output_file_path}{filename}{args.output_filename_addon}_params_{args.stage_name}.csv'
  pp = pd.DataFrame(params_data)
  pp.to_csv(params_filename, sep=',', index=False)

  return

def save_to_excel(filename, stats):
  filename = f'{filename}{args.output_filename_addon}.xlsx'
  workbook = xlsxwriter.Workbook(filename)
  if exists(filename):
    open_file_for_update(filename,workbook,args.excel_sheet_name)
  worksheet = workbook.add_worksheet(args.excel_sheet_name)
  format_head = workbook.add_format({'align': 'center'})
  format_cells = workbook.add_format({'align': 'center'})

  first = True
  idx_channel = 1
  for _,(_,s) in enumerate(stats.items()):
    for (channel_name, channel) in s.items():
      worksheet.write(idx_channel, 0, channel_name,format_head)
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



def save_final_to_excel(filename, stats):
  filename = f'{filename}{args.output_filename_addon}.xlsx'
  workbook = xlsxwriter.Workbook(filename)
  if exists(filename):
    open_file_for_update(filename,workbook,'Total')
  worksheet = workbook.add_worksheet('Total')
  format_head = workbook.add_format({'align': 'center'})
  format_cells = workbook.add_format({'align': 'center'})
  first = True
  idx_channel = 1
  for (channel_name, channel) in stats.items():
    worksheet.write(idx_channel, 0, channel_name,format_head)
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

def save_final_to_csv(filename, stats):
  data = {}
  for key, s in stats.items():
    data[key] = [s['TP'], s['FP'], s['FN'],s['PR'],s['RE'],s['F1']]

  filename = f'{args.output_file_path}{filename}{args.output_filename_addon}_final.csv'
  tp = pd.DataFrame(data)
  tp.to_csv(filename, sep=',', index=False)

  return

def open_file_for_update(filename,workbook,new_sheet_name):

  format_cells = workbook.add_format({'align': 'center'})
  wbRD  = xlrd.open_workbook(filename)
  sheets = wbRD.sheets()
  # run through the sheets and store sheets in workbook
  # this still doesn't write to the file yet
  for sheet in sheets: # write data from old file
    if sheet.name == new_sheet_name: #skip copy of sheet we will add
      continue
    newSheet = workbook.add_worksheet(sheet.name)
    for row in range(sheet.nrows):
        for col in range(sheet.ncols):
            v = sheet.cell(row, col).value
            newSheet.write(row, col, v, format_cells)
            if row == 0:
              newSheet.set_column(col, col, max(8, len(v) + 1))
            elif type(v) is str and v != "":
              newSheet.set_column(col, col, max(8, len(v) + 1))

  return

def get_channel_filenames(args,channels_list):
  # args_tmp = copy.deepcopy(args)
  run_file = open(f'run{args.output_filename_addon}.bat', 'r')
  lines = run_file.readlines()
  channel_filenames_list = []
  for channel_name in channels_list:
    for line in lines:
      line = line.rstrip(' \n')
      if channel_name in line:
        line_args = [x for x in line.split(' ')]
        if line_args[1] != 'swat_htm.py':
          continue
        line_args = line_args[2:]
        args_tmp = swat_htm.parser.parse_args(line_args)
        if args_tmp.channel_name != channel_name:
          continue

        fn = swat_utils.get_file_prefix(args_tmp, args_tmp.channel_name)
        print(f'{channel_name}:{fn}')
        channel_filenames_list.append(fn)
        break

  run_file.close()
  return channel_filenames_list

def main(args):
  json_filename = f'{args.output_file_path}{args.excel_filename}{args.output_filename_addon}.json'
  if args.final_stage:
    final_stage(json_filename)
    return
  else:
    if len(args.batch_channel_names) != 0:
      channels_list = [x for x in args.batch_channel_names.split(',')]
      channels_filenames_list = get_channel_filenames(args, channels_list)
    else:
      channels_list = [args.channel_name]
      channels_filenames_list = [swat_utils.get_file_prefix(args, args.channel_name)]

    first = True
    stats = {}
    channel_stats = {}
    for channel_name,file_prefix in zip(channels_list,channels_filenames_list):
      args.channel_name = channel_name
      if first:
        # label_prefix = swat_utils.get_file_prefix(args, args.channel_name)
        label_prefix = 'label'
        first = False
      stats[channel_name] = get_channel_stats(args,channel_name,file_prefix,label_prefix)
      channel_stats[channel_name] = stats[channel_name][channel_name]

    save_to_csv(args.excel_filename,stats)
    stats[args.stage_name] = {args.stage_name: unify_stats(channel_stats,args.grace_time)}

    save_for_final_stage(json_filename,args.stage_name,stats)

    del stats[args.stage_name][args.stage_name]['fp_array']

    for _, s in stats.items():
      for key, value in s.items():
        if 'fp_array' in value:
          del value['fp_array']
        print(f'{key:5}:{value}')


    save_to_excel(args.excel_filename, stats)

  return


def save_for_final_stage(filename, stage_name, stats):
  if os.path.isfile(filename):
    with open(filename) as f:
      final_stats = json.loads(f.read())
  else:
    final_stats = {}
  final_stats[stage_name] = stats[stage_name][stage_name]
  with open(filename, 'w') as fj:
    fj.write(json.dumps(final_stats))

  return

def final_stage(filename):
    with open(filename) as f:
      final_stats = json.loads(f.read())
    final_stats['Total'] = unify_stats(final_stats, args.grace_time)

    del final_stats['Total']['fp_array']

    save_final_to_excel(args.excel_filename, final_stats)
    save_final_to_csv(args.excel_filename, final_stats)
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

    # sys.argv = ['calc_anomaly_stats.py',
    #             '--stage_name', 'P1',
    #             '-bcn', 'LIT101,P102',
    #             '--freeze_type', 'off',
    #             '--learn_type', 'always',
    #             '-ofa', '_learn_mixed']

    # sys.argv = ['calc_anomaly_stats.py',
    #             '-ofa', '_learn_mixed','--final_stage']

    args = parser.parse_args()
    print(args)
    main(args)
