import sys
import argparse

import csv
import os
import numpy as np
import random
import math
import pandas
import swat_utils
from enum import Enum

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

parser = argparse.ArgumentParser(description='runtime configuration for HTM anomaly detection on SWAT')
parser.add_argument('--stage_name', '-sn', metavar='STAGE_NAME', default='P1', choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'], type=str.upper)
parser.add_argument('--channel_name', '-cn', metavar='CHANNEL_NAME')
parser.add_argument('--channel_type', '-ctype', metavar='CHANNEL_TYPE', default=0, type=int,help='set type 0 for analog, 1 for discrete')
parser.add_argument('--freeze_type', '-ft', default='off', choices=['off', 'during_training', 'end_training'], type=str.lower)
parser.add_argument('--learn_type', '-lt', default='always', choices=['always', 'train_only'], type=str.lower)
parser.add_argument('--sdr_size', '-size', metavar='SDR_SIZE', default=2048, type=int)
parser.add_argument('--connection_segments_gap', '-csg', default=1, type=int)
parser.add_argument('--sdr_sparsity', '-sparsity', metavar='SDR_SPARCITY', default=0.02, type=float)
parser.add_argument('--custom_min', '-cmin', metavar='MIN_VAL', default=2, type=int)
parser.add_argument('--custom_max', '-cmax', metavar='MAX_VAL', default=3, type=int)
parser.add_argument('--limits_enabled', '-le', default=False, action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--prefix', default="", type=str.lower)
parser.add_argument('--input_file_path', default="./HTM_input/", type=str)
parser.add_argument('--output_file_path', default="./HTM_results/", type=str)
parser.add_argument('--override_parameters', '-op', default="", type=str,
                    help="override parameter values, group_name,var_name,val,res/../.. ,param value = val/res")
parser.add_argument('--encoding_type', '-et', metavar='ENCODING_TYPE', default='diff', choices=['raw', 'diff'], type=str.lower)

default_parameters = {
    'enc': {
        "value" :
            {'size': 2048, 'sparsity': 0.02} #0.02
    },
    'predictor': {'sdrc_alpha': 0.1},
    'sp': {'boostStrength': 3.0,
           'localAreaDensity': 0.02,
           'potentialPct': 0.85,
           'synPermActiveInc': 0.04,
           'synPermConnected': 0.13999999999999999,
           'synPermInactiveDec': 0.006},
    'tm': {'activationThreshold': 8,
           'cellsPerColumn': 5,
           'initialPerm': 0.21,
           'maxSegmentsPerCell': 128,
           'maxSynapsesPerSegment': 96,
           'minThreshold': 3,
           'synPermConnected': 0.13999999999999999,
           'permanenceDec': 0.001,
            'cellNewConnectionMaxSegmentsGap': 0,
           'permanenceInc': 0.1},
    'anomaly': {'period': 1000},
}

def main(args):
    print('running ..')

    file_prefix = swat_utils.get_file_prefix(args,args.channel_name)
    output_filepath = ''.join([args.output_file_path, file_prefix])
    input_filepath = ''.join([args.input_file_path, args.stage_name, '_data.csv'])
    meta_filepath = ''.join([args.input_file_path, args.stage_name, '_meta.csv'])

    runtime_config = {'verbose': args.verbose,
                      'CustomMinMax': args.limits_enabled,
                      'CustomMin': args.custom_min,
                      'CustomMax': args.custom_max,
                      'learn_during_training_only': args.learn_type == "train_only",
                      'freeze_configuration': args.freeze_type,
                      'stage': args.stage_name,
                      'input_path': input_filepath,
                      'output_path': output_filepath,
                      'meta_path': meta_filepath,
                      'var_name': args.channel_name
                      }

    parameters = default_parameters
    parameters['enc']['size'] = args.sdr_size
    parameters['enc']['sparsity'] = args.sdr_sparsity
    parameters['tm']['cellNewConnectionMaxSegmentsGap'] = args.connection_segments_gap
    parameters['runtime_config'] = runtime_config

    if len(args.override_parameters) > 0:
        records_list = [item for item in args.override_parameters.split('/')]
        for record in records_list:
            param_list = [item for item in record.split(',')]
            if len(param_list) != 4:
                print(f"illegal param definition {param_list}")
                return
            else:
                print(f"override parameter: {param_list}")
                group_name = param_list[0]
                param_name = param_list[1]
                param_val = int(param_list[2])
                param_res = int(param_list[3])
                if param_res == 1:
                    parameters[group_name][param_name] = param_val
                else:
                    parameters[group_name][param_name] = float(param_val)/param_res

    runner(parameters,args)

def runner(parameters,args):
    config = parameters['runtime_config']
    verbose = config['verbose']
    stage = config['stage']
    learn_during_training_only = config['learn_during_training_only']
    freeze_trained_network = config['freeze_configuration'] == "end_training"
    freeze_during_training = config['freeze_configuration'] == "during_training"
    output_filepath = config['output_path']
    input_data = swat_utils.read_input(config['input_path'], config['meta_path'])
    assert stage.casefold() == input_data['stage'].casefold(), 'illegal input stage'

    training_count = input_data['training_count']
    features_info = input_data['features']
    records = input_data['records']

    if verbose:
        import pprint
        pprint.pprint(parameters, indent=4)
        prm_output_filepath = ''.join([output_filepath, '_param.txt'])
        with open(prm_output_filepath, 'w') as f:
            pprint.pprint(parameters, indent=4,stream=f)
            pprint.pprint(f"training points count: {training_count}", indent=4, stream=f)
            pprint.pprint(f"total points count: {len(records)}", indent=4, stream=f)
            pprint.pprint(features_info, indent=4, stream=f)

    print(f"training points count: {training_count}")
    print(f"total points count: {len(records)}")

    sdr_size = parameters["enc"]["size"]
    sdr_sparsity = parameters["enc"]["sparsity"]

    V1PrmName = config['var_name']
    V1EncoderParams = ScalarEncoderParameters()

    if config['CustomMinMax'] is True:
        print("Custom MinMax")
        V1EncoderParams.minimum = float(config['CustomMin'])
        V1EncoderParams.maximum = float(config['CustomMax'])
    else:
        V1EncoderParams.minimum = features_info[V1PrmName]['min']
        V1EncoderParams.maximum = features_info[V1PrmName]['max']

    if args.channel_type == 0:
        V1EncoderParams.size = sdr_size
        V1EncoderParams.sparsity = sdr_sparsity
    else:
        V1EncoderParams.category = 1

        V1EncoderParams.activeBits = int(sdr_size/(V1EncoderParams.maximum-V1EncoderParams.minimum+1))
        sdr_sparsity = float(V1EncoderParams.activeBits/sdr_size)
        print(f'active bits: {V1EncoderParams.activeBits}')

    print(f'min: {V1EncoderParams.minimum}, max: {V1EncoderParams.maximum}')
    V1Encoder = ScalarEncoder(V1EncoderParams)

    V1EncodingSize = V1Encoder.size
    encodingWidth = V1EncodingSize
    enc_info = Metrics( [encodingWidth], 999999999 )

    # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.


    # spParams = parameters["sp"]
    # sp = SpatialPooler(
    #     inputDimensions            = (encodingWidth,),
    #     columnDimensions           = (sdr_size,),
    #     potentialPct               = spParams["potentialPct"],
    #     potentialRadius            = encodingWidth,
    #     globalInhibition           = True,
    #     localAreaDensity           = spParams["localAreaDensity"],
    #     synPermInactiveDec         = spParams["synPermInactiveDec"],
    #     synPermActiveInc           = spParams["synPermActiveInc"],
    #     synPermConnected           = spParams["synPermConnected"],
    #     boostStrength              = spParams["boostStrength"],
    #     wrapAround                 = True
    # )
    # sp_info = Metrics( sp.getColumnDimensions(), 999999999 )

    tmParams = parameters["tm"]
    tm = TemporalMemory(
        columnDimensions          = (sdr_size,),
        cellsPerColumn            = tmParams["cellsPerColumn"],
        activationThreshold       = tmParams["activationThreshold"],
        initialPermanence         = tmParams["initialPerm"],
        connectedPermanence       = tmParams["synPermConnected"],
        minThreshold              = tmParams["minThreshold"],
        maxNewSynapseCount        = int(sdr_sparsity*sdr_size),
        permanenceIncrement       = tmParams["permanenceInc"],
        permanenceDecrement       = tmParams["permanenceDec"],
        cellNewConnectionMaxSegmentsGap=tmParams["cellNewConnectionMaxSegmentsGap"],
        predictedSegmentDecrement = 0.0,
        maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
        maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
    )
    tm_info = Metrics([tm.numberOfCells()], 999999999 )

    anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

    # Iterate through every datum in the dataset, record the inputs & outputs.

    N = len(records)
    inputs = [None]*N
    anomaly = [None]*N
    attack_label = [None]*(N-training_count)
    anomalyProb = [None]*N
    v1_idx = features_info[V1PrmName]['idx']
    attack_label_idx = features_info["Attack"]['idx']
    v1_prev = 2.4
    max_const_duration = 0
    current_const_duration = 0
    prev_encoding = 0
    test_count = 0
    for count, record in enumerate(records):

        v1_val = float(record[v1_idx])

        if v1_val > V1EncoderParams.minimum and v1_val < V1EncoderParams.maximum:
            v1_prev = v1_val

        if v1_val < V1EncoderParams.minimum:
            v1_val = v1_prev

        if v1_val > V1EncoderParams.maximum:
            v1_val = v1_prev

        inputs[count] = v1_val

        if count >= training_count:
            attack_label[count-training_count] = int(record[attack_label_idx])
            test_count += 1

        # Call the encoders to create bit representations for each value.  These are SDR objects.
        encoding = V1Encoder.encode(v1_val)
        enc_info.addData(encoding)

        learn = True
        if learn_during_training_only:
            learn = count < training_count

        permanent = False
        if freeze_during_training:
            permanent = count < training_count

        if freeze_trained_network and count == training_count:
            tm.make_current_network_permanent()
            print('training done, freeze network..')

        if args.encoding_type == 'raw':
            tm.compute(encoding, learn=learn, permanent=permanent)
            tm_active_cells = tm.getActiveCells()
            tm_info.addData( tm_active_cells.flatten() )
            anomaly[count] = tm.anomaly
            anomalyProb[count] = anomaly_history.compute(tm.anomaly)

        if args.encoding_type == 'diff':
            if count == 0 or prev_encoding != encoding:
                tm.compute(encoding, learn=learn, permanent=permanent)
                tm_active_cells = tm.getActiveCells()
                tm_info.addData(tm_active_cells.flatten())
                anomaly[count] = tm.anomaly
                anomalyProb[count] = anomaly_history.compute(tm.anomaly)
            else:
                anomaly[count] = 0
                anomalyProb[count] = 0

            prev_encoding = encoding

        # if count == 1:
        #     prev_value = v1_val;
        # else:
        #     if v1_val == prev_value:
        #         current_const_duration += 1
        #         if count > training_count:
        #             if current_const_duration > max_const_duration:
        #                 anomaly[count] = 1
        #     else:
        #         if count < training_count:
        #             max_const_duration = max(max_const_duration,current_const_duration)
        #
        #         current_const_duration = 0

        if count > 1 and count % 100000 == 0:
            print(f"{count}")
        if count < 5 or count % 10000 == 0:
            print(".", end=" ")

    if verbose:
        # Print information & statistics about the state of the HTM.
        print("Encoded Input", enc_info)
        print("")
        print("Temporal Memory Cells", tm_info)
        print(str(tm))
        print("")
        # Show info about the anomaly (mean & std)
        print("Anomaly Mean", np.mean(anomaly))
        print("Anomaly Std ", np.std(anomaly))

        prm_output_filepath = ''.join([output_filepath, '_stats.txt'])
        with open(prm_output_filepath, 'w') as f:
            pprint.pprint("Encoded Input", indent=4, stream=f)
            pprint.pprint(str(enc_info), indent=4, stream=f)
            pprint.pprint("", indent=4, stream=f)
            pprint.pprint("Temporal Memory", indent=4, stream=f)
            pprint.pprint(str(tm_info), indent=4, stream=f)
            pprint.pprint("", indent=4, stream=f)
            pprint.pprint(str(tm), indent=4, stream=f)
            pprint.pprint("", indent=4, stream=f)
            pprint.pprint(f"Anomaly Mean: {np.mean(anomaly)}", indent=4, stream=f)
            pprint.pprint(f"Anomaly STD: {np.std(anomaly)}", indent=4, stream=f)

    # placeholder..
    pred1 = anomaly
    pred5 = anomaly
    # end placeholder
    df = pandas.DataFrame(data={"Input": inputs, "1 Step Prediction": pred1, "5 Step Prediction": pred5,
                                "Anomaly Score": anomaly, "Anomaly Likelihood" : anomalyProb})
    htm_output_filepath = ''.join([output_filepath, '_res.csv']);
    df.to_csv(htm_output_filepath, sep=',', index=False)
    attack_label_output_filepath = ''.join([output_filepath, '_attack.real']);
    swat_utils.save_list(attack_label, attack_label_output_filepath)
    print(f'test_count: {test_count}, len: {len(attack_label)}')

    return

if __name__ == "__main__":
    # sys.argv = ['swat_htm.py',
    #             '--stage_name', 'P3',
    #             '--channel_name', 'DPIT301',
    #             '--freeze_type', 'off',
    #             '--learn_type', 'always',
    #             '--verbose',
    #             '-ctype','0']

    args = parser.parse_args()
    print(args)
    main(args)
