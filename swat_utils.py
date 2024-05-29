import csv
import pandas
from htm.bindings.sdr import SDR
import random

def read_input(input_path, meta_path, sampling_interval):
# read input for running HTM on SWAT

    meta = []
    records = []
    sampling_count = 0
    with open(input_path, "r") as fin:
        reader = csv.reader(fin)
        for record in reader:
            if sampling_count == 0:
              records.append(record)
            sampling_count +=1
            if sampling_count == sampling_interval:
              sampling_count = 0


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
             'training_count': int(meta[1])//sampling_interval,
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


def count_continuous_ones(data):
  prev_zero = True
  found = 0
  for d in data:
    if d:
      if prev_zero:
        prev_zero = False
        found = found + 1
    else:
      prev_zero = True

  return found


def calc_anomaly_stats(scores, labels, grace_time):
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
  label_grace_time = grace_time//10
  last_label_detected = False

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
      last_label_detected = False
      l_start_time = idx

    if l_now == False:
      in_label = False
      l_marked = False

    if in_label and l_marked == False and s_now:
      TP_detected_labels.append(l_count)
      TP_detection_delay.append(idx - l_start_time)
      last_label_detected = True
      l_marked = True


    if s_now and l_now:
      s_marked = True

    if (s_prev and s_now == False) or (s_now and idx == N-1):
      if s_marked == False:
        if not last_label_detected:
          max_hist = min(FP_start_idx, label_grace_time)
          for i in range(max_hist):
            if labels[FP_start_idx - i] == 1:
              TP_detected_labels.append(l_count)
              TP_detection_delay.append(idx - l_start_time)
              s_marked = True
              break

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
  stats['detected_labels'] = stage_id_to_global_id(0,TP_detected_labels)
  stats['detection_delay'] = TP_detection_delay
  assert len(TP_detection_delay) == len(stats['detected_labels']), 'TP_detection_delay len error'
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

def test_count_continuous_ones():
  a = [0,0,1,0,1,0,1,1,1,0,1,1,0,0,1,0]
  val = count_continuous_ones(a)
  print(f'found {val} continuous ones')

def computeAnomalyScore(active, predicted):
  if active.getSum() == 0:
    return 0.0

  both = SDR(active.dimensions)
  both.intersection(active, predicted)

  score = (active.getSum() - both.getSum()) / active.getSum()

  return score


def stage_id_to_global_id(stage_id, stage_id_list):
  # id:0 is all labels, id:1..6 is P1..P6
  stage_ids_map = [ [1, 2, 3, 6, 7, 8, 10, 11, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
                    [1, 2, 3, 21, 26,27, 28, 30, 33, 34, 35, 36],
                    [2, 6, 24, 26, 27, 28, 30],
                    [7, 8, 16, 17, 23, 26, 27, 28, 32, 41],
                    [8, 10, 11, 17, 22, 23, 25, 27, 28, 31, 37, 38, 39, 40],
                    [10, 11, 19, 20, 22,27, 28, 37, 38, 39, 40],
                    [8, 23, 28]
                    ]

  assert stage_id >= 0 and  stage_id <=6, 'illegal stage id'

  res = [stage_ids_map[stage_id][idx-1] for idx in stage_id_list]
  return res

def get_delay_sdr_width(bin_size : int):
  assert bin_size >= 1 and bin_size <= 20, 'bin size > 20'
  if bin_size == 1:
    return 1
  if bin_size == 2:
    return 2
  if bin_size == 3:
    return 3
  if bin_size >= 4 and bin_size <=6:
    return 4
  if bin_size >= 7 and bin_size <=10:
    return 5
  if bin_size >= 11 and bin_size <=20:
    return 6


def get_delay_bin_idx(bins, value):
  for idx ,bin_val in enumerate(bins):
    if value < bin_val:
      return idx

  return len(bins)


def get_delay_active_columns_num(sdr_len):
  assert sdr_len >=0 and sdr_len <= 6, 'illegal sdr_len'
  val = [0, 1, 1, 2, 2, 2, 3]
  return val[sdr_len]


def get_delay_sdr(state_idx, sdr_len):

  s = [[[1]], #1
       [[1, 0], #2
        [0, 1]],
       [[1, 1, 0], #3
        [1, 0, 1],
        [0, 1, 1]],
       [[1, 1, 0, 0], #4
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1]],
       [[1, 1, 0, 0, 0], #5
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1]],
     [[1, 1, 1, 0, 0, 0], #6
      [1, 1, 0, 1, 0, 0],
      [1, 1, 0, 0, 1, 0],
      [1, 1, 0, 0, 0, 1],
      [1, 0, 1, 1, 0, 0],
      [1, 0, 1, 0, 1, 0],
      [1, 0, 1, 0, 0, 1],
      [1, 0, 0, 1, 1, 0],
      [1, 0, 0, 1, 0, 1],
      [1, 0, 0, 0, 1, 1],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 1, 0, 1, 0],
      [0, 1, 1, 0, 0, 1],
      [0, 1, 0, 1, 1, 0],
      [0, 1, 0, 1, 0, 1],
      [0, 1, 0, 0, 1, 1],
      [0, 0, 1, 1, 1, 0],
      [0, 0, 1, 1, 0, 1],
      [0, 0, 1, 0, 1, 1],
      [0, 0, 0, 1, 1, 1]]]

  assert sdr_len <= 6 , 'get_state_sdr: sdr_len >  6'
  assert state_idx < len(s[sdr_len-1]), 'get_state_sdr: idx > max len'

  return s[sdr_len-1][state_idx]


def test_get_state_sdr():
  s = get_delay_sdr(0,2)
  assert s == [1,0], 'get_state_sdr(0,2)'

  s = get_delay_sdr(1,3)
  assert s == [1, 0, 1], ' get_state_sdr(1,3)'

  s = get_delay_sdr(19,6)
  assert s == [0, 0, 0, 1, 1, 1], ' get_state_sdr(19,6)'

  print('test_get_state_sdr test done')

def test_stage_id_to_global_id():
  ids = stage_id_to_global_id(1,[1,4,11])
  assert ids == [1,21,36], 'mapping error'

def and_blist(x, y):
    return [a and b for a, b in zip(x, y)]


def or_blist(x, y):
  return [a or b for a, b in zip(x, y)]


def not_blist(x):
  return [not a for a in x]


def list2blist(x):
  return [a != 0 for a in x]


def blist2list(x):
  return [int(a) for a in x]


def SDR2blist(x):
  val = [False] * x.size
  for i in x.sparse:
    val[i] = True

  return val


def blist2SDR(x):
  res = SDR(len(x))
  res.sparse = [i for i, val in enumerate(x) if x[i]]

  return res


def stable_cdt(SDRT, target_sparsity, permutation, alpha=1.1):
  # assume SDR is binary list
  type = 0
  if type == 1:
    rng = random.Random()
    rng.seed(100)
  else:
    idx_perm = 0

  #    SDRT = list2blist(bSDR)
  N = len(SDRT)
  SDR_FINAL = [False] * N
  PKZ = list(SDRT)

  NK0 = 0
  NK1 = 0

  while (sum(SDR_FINAL) / N < target_sparsity):
    if type == 1:
      rng.shuffle(PKZ)
    else:
      PKZ[:] = [PKZ[i] for i in permutation[idx_perm]]
      idx_perm = 1 if idx_perm == 0 else 0

    SDR_FINAL = or_blist(SDR_FINAL, and_blist(SDRT, PKZ))
    NK1 = NK1 + 1

  #print(f"sparsity end of additive {sum(SDR_FINAL) / N}")

  while (sum(SDR_FINAL) / N > target_sparsity * alpha):
    if type == 1:
      rng.shuffle(PKZ)
    else:
      PKZ[:] = [PKZ[i] for i in permutation[idx_perm]]
      idx_perm = 1 if idx_perm == 0 else 0

    SDR_FINAL = and_blist(SDR_FINAL, not_blist(PKZ))
    NK0 = NK0 + 1

  #print(f"sparsity end of substructive {sum(SDR_FINAL) / N}")

  return blist2list(SDR_FINAL), NK0, NK1


def encode_sequence(SDR_SEQ, permutation):
  # assume SDR_SEQ is binary list
  #    rng = random.Random()
  #    rng.seed(seed_val)
  N = len(SDR_SEQ[0])
  #    permutation = rng.shuffle(list(range(N)))

  SDR_FINAL = [False] * N

  # for 3 sdrs the final sdr is sdr[0] + sdr[1]*p + sdr[2]*p*p
  for idx, sdr in enumerate(SDR_SEQ):
    for i in range(idx+1):
      sdr[:] = [sdr[j] for j in permutation]

    SDR_FINAL = or_blist(SDR_FINAL, sdr)

  return SDR_FINAL

def test_cdt():
  sdr_val = list()
  sdr_bin_list = list()
  rng = random.Random()
  N = 2048
  bits = 41
  rng.seed(10)
  sparsity = 0.02
  permutation_cdt = list()
  permutation_enc = list(range(N))
  rng.shuffle(permutation_enc)
  permutation_cdt.append(list(range(N)))
  permutation_cdt.append(list(range(N)))
  rng.shuffle(permutation_cdt[0])
  rng.shuffle(permutation_cdt[1])

  for i in range(16):
    sdr_val.append(SDR(N))
    sdr_val[i].sparse = list(range(20 + i, 61 + i))

  for i in range(16):
    sdr_bin_list.append(SDR2blist(sdr_val[i]))

  sdr_encoded_bin = encode_sequence(sdr_bin_list, permutation_enc)



  sdr_encoded = blist2SDR(sdr_encoded_bin)
  sdr_cdt_bin,N0,N1 = stable_cdt(sdr_encoded_bin, sparsity, permutation_cdt)

  sdr_cdt = blist2SDR(sdr_cdt_bin)

  print(f"size: {sdr_cdt.size}, N0 {N0}, {N1}")


def main():
  test_cdt()
  print("running..")

if __name__ == "__main__":
  #test_stage_id_to_global_id()

  # test_get_state_sdr()
  # test_calc_anomaly_stats()
  #test_count_continuous_ones()
  main()

