#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
NUMBER_OF_REPEATS=20
bash byFreq/create_byFreq_x_times.sh $NUMBER_OF_REPEATS
bash byMaterialThickness/create_byMaterialThickness_x_times.sh $NUMBER_OF_REPEATS
bash byPastCoeff/create_byPastCoeff_x_times.sh $NUMBER_OF_REPEATS
bash bySensitivity/create_bySensitivity_x_times.sh $NUMBER_OF_REPEATS
bash bySignalNoise/create_bySignalNoise_x_times.sh $NUMBER_OF_REPEATS
bash multilateration_and_trajectory/create_Multilateration_x_times.sh $NUMBER_OF_REPEATS
bash multiplePOITracking/create_multiplePOI_figs.sh
bash realWorldEnvironment_Experiments/runRealWorldExperimentScripts.sh
bash receiverPositioning/create_receiverPositioning_figs.sh
bash Wi-Fi_test/create_Wi-Fi_test_figs.sh