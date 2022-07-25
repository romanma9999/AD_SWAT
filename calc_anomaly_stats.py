import sys
import swat_utils
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='runtime configuration for HTM anomaly statistics on SWAT')
parser.add_argument('--stage_name', '-sn', metavar='STAGE_NAME', default='P1', choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'], type=str.upper)
parser.add_argument('--channel_name', '-cn', metavar='CHANNEL_NAME',type=str)
parser.add_argument('--batch_channel_names', '-bcn', metavar='BATCH_CHANNEL_NAMES',default = "",type=str)
parser.add_argument('--freeze_type', '-ft', default='off', choices=['off', 'during_training', 'end_training'], type=str.lower)
parser.add_argument('--learn_type', '-lt', default='always', choices=['always', 'train_only'], type=str.lower)
parser.add_argument('--prefix', default="", type=str.lower)
parser.add_argument('--output_file_path', default="./HTM_results/", type=str)
parser.add_argument('--raw_threshold', '-rth', default=0.6, type=float, help="raw anomaly score threshold")
parser.add_argument('--logl_threshold', '-lglth', default=0.6, type=float, help="log likelihood anomaly score threshold")
parser.add_argument('--mean_window', '-rw', default=9, type=int, help="moving mean anomaly score window")
parser.add_argument('--mean_threshold', '-mth', default=0.6, type=float, help="moving mean anomaly score threshold")
parser.add_argument('--sum_window', '-sw', default=119, type=int, help="moving sum anomaly score window")
parser.add_argument('--sum_threshold', '-sth', default=0.6, type=float, help="moving sum anomaly score threshold")
parser.add_argument('--training_count', '-tc', default=414000, type=int, help="training points count")

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

def get_channel_stats(args,channel_name):
  print(f'process {channel_name}.')
  file_prefix = swat_utils.get_file_prefix(args, args.channel_name)
  input_filepath = ''.join([args.output_file_path, file_prefix, '_res.csv'])
  label_filepath = ''.join([args.output_file_path, file_prefix, '_attack.real'])

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
  stats['raw'] = process_score(labels, raw_scores, 'raw', args.raw_threshold, args.output_file_path, file_prefix)
  stats['logl'] = process_score(labels, logl_scores, 'logl', args.logl_threshold, args.output_file_path, file_prefix)
  stats['mean'] = process_score(labels, mean_scores, 'mean', args.mean_threshold, args.output_file_path, file_prefix)
  stats['sum'] = process_score(labels, sum_scores, 'sum', args.sum_threshold, args.output_file_path, file_prefix)

  max_stats = {}
  max_key = get_key_with_max_F1(stats)
  max_stats[max_key] = stats.pop(max_key)
  print(f'{max_key} score selected, F1 = {max_stats[max_key]["F1"]}')
  max_key = get_key_with_max_F1(stats)
  max_stats[max_key] = stats.pop(max_key)
  print(f'{max_key} score selected, F1 = {max_stats[max_key]["F1"]}')

  total_stats = unify_stats(max_stats)

  for key, value in max_stats.items():
    print(f'{key:5}:{value}')

  return total_stats

def main(args):
  if len(args.batch_channel_names) != 0:
    channels_list = [x for x in args.batch_channel_names.split(',')]
  else:
    channels_list = [args.channel_name]

  stats = {}
  for channel_name in channels_list:
    args.channel_name = channel_name
    stats[channel_name] = get_channel_stats(args,channel_name)

  total_stats = unify_stats(stats)
  for key, value in stats.items():
    print(f'{key:5}:{value}')

  del total_stats['fp_array']
  print(f'{args.stage_name:5}:{total_stats}')

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
                '--freeze_type', 'off',
                '--learn_type', 'always',
                '--raw_threshold', '0.7']

    args = parser.parse_args()
    print(args)
    main(args)
