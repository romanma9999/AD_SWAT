import csv
import os
import numpy as np
import random
import math
import pandas

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

# _EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
# _INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, _INPUT_FILE_NAME)
# _OUTPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, _OUTPUT_FILE_NAME)
_INPUT_FILE_PATH = "./HTM_input/";
_OUTPUT_FILE_PATH = "./HTM_results/";

sdr_size = 102;
sdr_w = 2;
sdr_sparsity = 1.01*(float(sdr_w)/sdr_size)

default_parameters = {
  'enc': {
    "value" :
      {'size': sdr_size, 'sparsity': sdr_sparsity} #0.02
  },
  'predictor': {'sdrc_alpha': 0.1},
  'sp': {'boostStrength': 3.0,
         'columnCount': sdr_size,    #1638
         'localAreaDensity': 0.04395604395604396,
         'potentialPct': 0.85,
         'synPermActiveInc': 0.04,
         'synPermConnected': 0.13999999999999999,
         'synPermInactiveDec': 0.006},
  'tm': {'activationThreshold': 1,
         'cellsPerColumn': 5,
         'initialPerm': 0.21,
         'maxSegmentsPerCell': 128,
         'maxSynapsesPerSegment': 96,
         'minThreshold': 1,
         'newSynapseCount': sdr_w,
         'permanenceDec': 0.1,
         'permanenceInc': 0.1},
  'anomaly': {'period': 1000},
}

def main(batch_param, filename_prefix, parameters=default_parameters, argv=None, verbose=True):
  input_filepath = ''.join([_INPUT_FILE_PATH, filename_prefix, '_data.csv'])
  meta_filepath = ''.join([_INPUT_FILE_PATH, filename_prefix, '_meta.csv'])
  if batch_param:
    group_name = batch_param["group_name"];
    param_name = batch_param["param_name"];
    param_res = batch_param["values_res"];
    for value in batch_param["values"]:
      if param_res != 1:
        parameters[group_name][param_name] = float(value)/param_res;
      else:
        parameters[group_name][param_name] = value;
      output_filepath = ''.join([_OUTPUT_FILE_PATH, filename_prefix, '_', param_name, '_', str(value), '.csv'])
      runner(parameters=parameters, verbose=verbose,input_filepath = input_filepath,meta_filepath=meta_filepath,output_filepath = output_filepath)
  else:
    output_filepath = ''.join([_OUTPUT_FILE_PATH, filename_prefix, '_res.csv'])
    runner(parameters=parameters, verbose=verbose,input_filepath = input_filepath,meta_filepath=meta_filepath,output_filepath = output_filepath)


def runner(parameters, verbose,input_filepath,meta_filepath,output_filepath):

  if verbose:
    import pprint
    print("Parameters:")
    pprint.pprint(parameters, indent=4)
    print("")


  # Read the input file.
  meta = []
  records = []
  with open(input_filepath, "r") as fin:
    reader = csv.reader(fin)
    for record in reader:
      records.append(record);

  with open(meta_filepath, "r") as fin:
    reader = csv.reader(fin)
    for record in reader:
      meta.extend(record);

  training_count = int(meta[1]);
  print(f"running SWAT stage {meta[0]}");
  print(f"features number {meta[2]}");
  print(f"training points count {training_count}");

  features_info = dict()
  for idx in range(int(meta[2])):
    key = str(meta[3 + 3 * idx]);
    min_val = int(meta[3 + 3 * idx + 1]);
    max_val = int(meta[3 + 3 * idx + 2]);
    features_info.update({key: [min_val, max_val]});

  scalarEncoderParams = ScalarEncoderParameters()
  scalarEncoderParams.category = 1;
  scalarEncoderParams.activeBits = 2;
  scalarEncoderParams.minimum = 0;
  scalarEncoderParams.maximum = 50;
  scalarEncoder = ScalarEncoder(scalarEncoderParams)  # 'scalar encoder'
  encodingWidth = scalarEncoder.size

  enc_info = Metrics( [encodingWidth], 999999999 )

  tmParams = parameters["tm"]
  columnCount = parameters["sp"]["columnCount"];
  tm = TemporalMemory(
    columnDimensions          = (columnCount,),
    cellsPerColumn            = tmParams["cellsPerColumn"],
    activationThreshold       = tmParams["activationThreshold"],
    initialPermanence         = tmParams["initialPerm"],
    connectedPermanence       = parameters["sp"]["synPermConnected"],
    minThreshold              = tmParams["minThreshold"],
    maxNewSynapseCount        = tmParams["newSynapseCount"],
    permanenceIncrement       = tmParams["permanenceInc"],
    permanenceDecrement       = tmParams["permanenceDec"],
    predictedSegmentDecrement = 0.0,
    maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
  )
  tm_info = Metrics( [tm.numberOfCells()], 999999999 )

  anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

  predictor = Predictor( steps=[1, 5], alpha=parameters["predictor"]['sdrc_alpha'] )
  predictor_resolution = scalarEncoder.parameters.resolution

  # Iterate through every datum in the dataset, record the inputs & outputs.


  anomaly = []
  anomalyProb = []
  predictions = {1: [], 5: []}
  learn_during_training_only = False
  LIT101_idx = 0;
  MV101_idx = 1;
  P101_idx = 2;
  P102_idx = 3;
  CombinedState_idx = 4;

  # inputs = [0,1,0,1,0,1,0,1,0,1,0,1,0,1];
  # inputs = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
  inputs = [0,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
  # inputs = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
  zeros_SDR = SDR(98)
  for count, value in enumerate(inputs):

      encoding = scalarEncoder.encode(value)
      enc_info.addData( encoding )
      tm.compute(encoding, learn=True);
      tm_active_cells = tm.getActiveCells();
      tm_info.addData( tm_active_cells.flatten() );

      anomaly.append( tm.anomaly )
      anomalyProb.append( anomaly_history.compute(tm.anomaly) )


  # Print information & statistics about the state of the HTM.
  print("Encoded Input", enc_info)
  print("")
  # print("Spatial Pooler Mini-Columns", sp_info)
  # print(str(sp))
  # print("")
  print("Temporal Memory Cells", tm_info)
  print(str(tm))
  print("")

  # Shift the predictions so that they are aligned with the input they predict.
  for n_steps, pred_list in predictions.items():
    for x in range(n_steps):
      pred_list.insert(0, float('nan'))
      pred_list.pop()

  # Calculate the predictive accuracy, Root-Mean-Squared
  accuracy         = {1: 0, 5: 0}
  accuracy_samples = {1: 0, 5: 0}

  for idx, inp in enumerate(inputs):
    for n in predictions: # For each [N]umber of time steps ahead which was predicted.
      val = predictions[n][ idx ]
      if not math.isnan(val):
        accuracy[n] += (inp - val) ** 2
        accuracy_samples[n] += 1
  for n in sorted(predictions):
    accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** .5
    print("Predictive Error (RMS)", n, "steps ahead:", accuracy[n])

  # Show info about the anomaly (mean & std)
  print("Anomaly Mean", np.mean(anomaly))
  print("Anomaly Std ", np.std(anomaly))

  df = pandas.DataFrame(data={"Input": inputs, "1 Step Prediction": predictions[1], "5 Step Prediction": predictions[5],
                              "Anomaly Score": anomaly, "Anomaly Likelihood" : anomalyProb})
  df.to_csv(output_filepath, sep=',', index=False)

  return -accuracy[5]



if __name__ == "__main__":
  print('running ..')
  batch_param = {'group_name':'tm','param_name':'permanenceDec', 'values' : [70,80,90,100], 'values_res':1000};
  filename_prefix = 'P1';
  main(batch_param = [], filename_prefix = filename_prefix)
