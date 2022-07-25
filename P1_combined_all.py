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

sdr_size = 1470;
sdr_w = 65;
sdr_sparsity = 1.01*(float(sdr_w)/sdr_size)

default_parameters = {
    'enc': {
        "value" :
            {'size': sdr_size, 'sparsity': sdr_sparsity} #0.02
    },
    'predictor': {'sdrc_alpha': 0.1},
    'sp': {'boostStrength': 3.0,
           'columnCount': sdr_size,    #1638
           'localAreaDensity': 0.02,
           'potentialPct': 0.85,
           'synPermActiveInc': 0.04,
           'synPermConnected': 0.13999999999999999,
           'synPermInactiveDec': 0.006},
    'tm': {'activationThreshold': 30,
           'cellsPerColumn': 5,
           'initialPerm': 0.21,
           'maxSegmentsPerCell': 128,
           'maxSynapsesPerSegment': 96,
           'minThreshold': 10,
           'newSynapseCount': sdr_w,
           'permanenceDec': 0.001,
           'permanenceInc': 0.1},
    'anomaly': {'period': 300},
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
            runner(parameters=parameters, verbose=verbose,input_filepath = input_filepath,meta_filepath=meta_filepath,output_filepath = output_filepath,stage = filename_prefix)
    else:
        output_filepath = ''.join([_OUTPUT_FILE_PATH, filename_prefix, '_res.csv'])
        runner(parameters=parameters, verbose=verbose,input_filepath = input_filepath,meta_filepath=meta_filepath,output_filepath = output_filepath,stage = filename_prefix)


def runner(parameters, verbose,input_filepath,meta_filepath,output_filepath,stage):


    if verbose:
        import pprint
        print("Parameters:")
        pprint.pprint(parameters, indent=4)
        prm_output_filepath = ''.join([output_filepath, '_parameters.txt'])
        with open(prm_output_filepath, 'w') as f:
            pprint.pprint(parameters, indent=4,stream=f)
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

    if stage.casefold() == meta[0].casefold():
        print(f"running SWAT stage {meta[0]}");
    else:
        print(f"error: input file stage is {meta[0]} and defined stage is {stage}")
        return

    print(f"features number {meta[2]}");
    print(f"training points count {training_count}");

    features_info = dict()
    for idx in range(int(meta[2])):
        key     = str(meta[3+3*idx]);
        min_val = int(meta[3+3*idx+1]);
        max_val = int(meta[3+3*idx+2]);
        features_info.update({key : {'min' : min_val, 'max' : max_val}});

    print(f"features info {features_info}")
    LITEncoderParams            = ScalarEncoderParameters()
    LITEncoderParams.size       = 1400
    LITEncoderParams.sparsity   = 0.03
    LITEncoderParams.minimum = features_info['LIT101']['min']
    LITEncoderParams.maximum = features_info['LIT101']['max']
    LITEncoder = ScalarEncoder( LITEncoderParams )

    P101_StateEncoderParams            = ScalarEncoderParameters()
    P101_StateEncoderParams.category = 1
    P101_StateEncoderParams.activeBits = 10
    P101_StateEncoderParams.minimum = features_info['P101']['min']
    P101_StateEncoderParams.maximum = features_info['P101']['max']
    P101_StateEncoder = ScalarEncoder( P101_StateEncoderParams )

    P102_StateEncoderParams = ScalarEncoderParameters()
    P102_StateEncoderParams.category = 1
    P102_StateEncoderParams.activeBits = 10
    P102_StateEncoderParams.minimum = features_info['P102']['min']
    P102_StateEncoderParams.maximum = features_info['P102']['max']
    P102_StateEncoder = ScalarEncoder(P102_StateEncoderParams)

    MV101_StateEncoderParams = ScalarEncoderParameters()
    MV101_StateEncoderParams.category = 1
    MV101_StateEncoderParams.activeBits = 10
    MV101_StateEncoderParams.minimum = features_info['MV101']['min']
    MV101_StateEncoderParams.maximum = features_info['MV101']['max']
    MV101_StateEncoder = ScalarEncoder(MV101_StateEncoderParams)

    P101_StateEncodingSize = P101_StateEncoder.size
    P102_StateEncodingSize = P102_StateEncoder.size
    MV101_StateEncodingSize = MV101_StateEncoder.size
    LITEncodingSize = LITEncoder.size
    encodingWidth = (P101_StateEncodingSize + P102_StateEncodingSize + MV101_StateEncodingSize + LITEncodingSize)
    enc_info = Metrics( [encodingWidth], 999999999 )

    # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
    spParams = parameters["sp"]
    sp = SpatialPooler(
        inputDimensions            = (encodingWidth,),
        columnDimensions           = (spParams["columnCount"],),
        potentialPct               = spParams["potentialPct"],
        potentialRadius            = encodingWidth,
        globalInhibition           = True,
        localAreaDensity           = spParams["localAreaDensity"],
        synPermInactiveDec         = spParams["synPermInactiveDec"],
        synPermActiveInc           = spParams["synPermActiveInc"],
        synPermConnected           = spParams["synPermConnected"],
        boostStrength              = spParams["boostStrength"],
        wrapAround                 = True
    )
    sp_info = Metrics( sp.getColumnDimensions(), 999999999 )

    tmParams = parameters["tm"]
    tm = TemporalMemory(
        columnDimensions          = (parameters["sp"]["columnCount"],),
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
    predictor_resolution = LITEncoder.parameters.resolution

    # Iterate through every datum in the dataset, record the inputs & outputs.
    inputs = []
    anomaly     = []
    anomalyProb = []
    predictions = {1: [], 5: []}
    learn_during_training_only = False
    LIT101_idx = 0;
    MV101_idx = 1;
    P101_idx = 2;
    P102_idx = 3;
    CombinedState_idx = 4;

    for count, record in enumerate(records):
        litVal = float(record[LIT101_idx])
        inputs.append(litVal)


        P101_stateVal = int(record[P101_idx])
        P102_stateVal = int(record[P102_idx])
        MV101_stateVal = int(record[MV101_idx])


        # Call the encoders to create bit representations for each value.  These are SDR objects.
        litBits = LITEncoder.encode(litVal)
        P101_stateBits = P101_StateEncoder.encode(P101_stateVal)
        P102_stateBits = P102_StateEncoder.encode(P102_stateVal)
        MV101_stateBits = MV101_StateEncoder.encode(MV101_stateVal)

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR(encodingWidth).concatenate([litBits, P101_stateBits, P102_stateBits, MV101_stateBits])
        enc_info.addData(encoding)

        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        #activeColumns = SDR(sp.getColumnDimensions())

        #sp_learning_enabled = count < training_count/4
        #sp_learning_enabled = True
        #tm_learning_enabled = True
        #if learn_during_training_only == 1:
        #    tm_learning_enabled = count < training_count


        # Execute Spatial Pooling algorithm over input space.
        #sp.compute(encoding, sp_learning_enabled, activeColumns)
        # Execute Temporal Memory algorithm over active mini-columns.
        #if sp_learning_enabled == False:
        #tm.compute(activeColumns, tm_learning_enabled)
        tm.compute(encoding, True)

        tm_active_cells = tm.getActiveCells()

        #sp_info.addData(activeColumns)
        tm_info.addData( tm_active_cells.flatten() )

        # Predict what will happen, and then train the predictor based on what just happened.
        #pdf = predictor.infer(tm_active_cells )

        # for n in (1, 5):
        #   if pdf[n]:
        #     predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution + LITEncoderParams.minimum)
        #   else:
        #     predictions[n].append(float('nan'))

        for n in (1, 5):
            predictions[n].append(1)

        anomaly.append(tm.anomaly )
        anomalyProb.append( anomaly_history.compute(tm.anomaly) )

        #predictor.learn(count, tm_active_cells, int((litVal-LITEncoderParams.minimum) / predictor_resolution))
        if count > 1 and count % 100000 == 0:
            print(f"{count}")
        if count < 5 or count % 10000 == 0:
            print(".", end=" ")

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

    # Plot the Predictions and Anomalies.
    if verbose:
        try:
            import matplotlib.pyplot as plt
        except:
            print("WARNING: failed to import matplotlib, plots cannot be shown.")
            return -accuracy[5]

        df = pandas.DataFrame(data={"Input": inputs, "1 Step Prediction": predictions[1], "5 Step Prediction": predictions[5],
                                    "Anomaly Score": anomaly, "Anomaly Likelihood" : anomalyProb})
        htm_output_filepath = ''.join([output_filepath, '.csv']);
        df.to_csv(htm_output_filepath, sep=',', index=False)

    return -accuracy[5]



if __name__ == "__main__":
    print('running ..')
    batch_param = {'group_name':'tm','param_name':'permanenceDec', 'values' : [70,80,90,100], 'values_res':1000};
    filename_prefix = 'P1';
    main(batch_param = [], filename_prefix = filename_prefix)
