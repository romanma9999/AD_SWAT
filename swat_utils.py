import csv

def read_input(input_path, meta_path):
# read input for running HTM on SWAT

    meta = []
    records = []
    with open(input_path, "r") as fin:
        reader = csv.reader(fin)
        for record in reader:
            records.append(record)

    with open(meta_path, "r") as fin:
        reader = csv.reader(fin)
        for record in reader:
            meta.extend(record)

    features_info = dict()
    for idx in range(int(meta[2])):
        pos = 3+3*idx
        features_info.update({str(meta[pos]): {'idx': idx, 'min': int(meta[pos+1]), 'max': int(meta[pos+2])}})

    input_data = {'meta': meta,
             'records': records,
             'stage': meta[0],
             'training_count': int(meta[1]),
             'features': features_info}

    return input_data


def get_file_prefix(args,channel_name):
  file_prefix = args.stage_name
  file_prefix += '_'
  file_prefix += channel_name
  file_prefix += "_learn_"
  file_prefix += args.learn_type
  file_prefix += "_freeze_"
  file_prefix += args.freeze_type
  if args.prefix != "":
    file_prefix += '_'
    file_prefix += args.prefix

  return file_prefix

def save_list(data, output_path):
  with open(output_path, 'w') as fp:
    for item in data:
      fp.write("%s\n" % item)

  return

def anomaly_score(data, threshold):
  res = [None]*len(data)
  for idx, item in enumerate(data):
    res[idx] = 1 if item >= threshold else 0

  return res


def calc_anomaly_stats(scores, labels, grace_time = 60):
  l_count = 0
  s_false_count = 0
  TP_detected_labels = []
  TP_detection_delay = []
  l_start_time = 0
  N = len(labels)
  FP_arr = [0]*N
  FP_start_idx = 0
  stats = {}
  stats['TP'] = 0
  stats['FP'] = 0
  stats['FN'] = 0
  stats['PR'] = 0.0
  stats['RE'] = 0.0
  stats['F1'] = 0.0
  stats['detected_labels'] = []
  stats['detection_delay'] = []
  stats['fp_array'] = []
  stats['LabelsCount'] = 0


  if len(scores) != N:
    print(f'Error, labels{N} and anomaly{len(scores)} vectors length is different')
    return stats

  l_prev = False
  s_prev = False
  l_now = False
  s_now = False

  l_marked = False
  in_label = False
  s_marked = False
  start = True
  for idx, (score,label) in enumerate(zip(scores,labels)):
    #skip score tail from training..
    if start and label == 0 and score == 1:
      continue
    start = False

    l_prev = l_now
    l_now = True if label == 1 else False
    s_prev = s_now
    s_now = True if score == 1 else False

    if s_now and s_prev == False:
      FP_start_idx = idx

    if l_now and l_prev == False:
      l_count = l_count+1
      in_label = True
      l_start_time = idx

    if l_now == False:
      in_label = False
      l_marked = False

    if in_label and l_marked == False and s_now:
      TP_detected_labels.append(l_count)
      TP_detection_delay.append(idx - l_start_time)
      l_marked = True


    if s_now and l_now:
      s_marked = True

    if (s_prev and s_now == False) or (s_now and idx == N-1):
      if s_marked == False:
        max_hist = min(FP_start_idx, grace_time)
        for i in range(max_hist):
          if FP_arr[FP_start_idx - i] == 1 or labels[FP_start_idx - i] == 1:
            s_marked = True
            break

      if s_marked == False:
        s_false_count = s_false_count + 1
        FP_arr[FP_start_idx:idx] = [1]*(idx-FP_start_idx)
        if s_now and idx == N-1:
          FP_arr[-1] = 1
      s_marked = False



  TP = len(TP_detected_labels)
  FN = l_count - TP
  FP = s_false_count
  PR = precision(TP, FP)
  RE = recall(TP, FN)

  stats = {}
  stats['TP'] = TP
  stats['FP'] = FP
  stats['FN'] = FN
  stats['PR'] = PR
  stats['RE'] = RE
  stats['F1'] = F1(PR, RE)
  stats['detected_labels'] = TP_detected_labels
  stats['detection_delay'] = TP_detection_delay
  stats['fp_array'] = FP_arr
  stats['LabelsCount'] = l_count

  return stats


def precision(TP,FP):
  return TP / (TP + FP) if TP + FP != 0 else 0.0

def recall(TP,FN):
  return TP / (TP + FN) if TP + FN != 0 else 0.0

def F1(PR,RE):
  return 2*PR*RE/(PR+RE) if PR+RE != 0 else 0.0


def test_calc_anomaly_stats():
  #1
  l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  idx = 0
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  # 2
  l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  s = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #3
  l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  s = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #4
  l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  s = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #5
  l = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  s = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #6
  l = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  s = [0, 0, 0, 1, 0, 1, 0, 1, 1, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #7
  l = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  s = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #8
  l = [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0]
  s = [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #9
  l = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
  s = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #10
  l = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
  s = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #11
  l = [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1]
  s = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

  #12
  l = [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1]
  s = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
  idx = idx + 1
  stats = calc_anomaly_stats(s, l)
  print(f'{idx}: {stats}')

if __name__ == "__main__":
  test_calc_anomaly_stats()