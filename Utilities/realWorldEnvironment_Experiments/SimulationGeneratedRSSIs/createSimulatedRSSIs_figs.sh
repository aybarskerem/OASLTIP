#!/bin/bash

echo -e "RUNNING REAL WORLD EXPERIMENTS WITH SIMULATED RSSIs...\n"
cd "$(dirname "${BASH_SOURCE[0]}")"

echo -e "Removing previously created  mean and #of no signal files...\n"
rm -f {1619_mu,1707_mu,1841_mu,numberOfNoSignals}.txt

rm ScriptsForAllCases/1619_InPocket_crowded/{1619_mu.txt,numberOfNoSignals.txt}
rm ScriptsForAllCases/1707_InHand_Crowded/{1707_mu.txt,numberOfNoSignals.txt}
rm ScriptsForAllCases/1841_InPocket_NonCrowded/{1841_mu.txt,numberOfNoSignals.txt}

echo -e "Removing means_avgAcc and stds_avgAcc files previously created by calculate_mean_std_realworld scripts...\n"
rm -f means_avgAcc.txt
rm -f stds_avgAcc.txt

echo -e "Removing image files...\n"
rm {realWorld_simulation_avgAcc,realworld_simulation_noSignal,realWorld_simulation_accuracies,CDF_Errors_Simulation}.png

echo "Running In-Pocket, Crowded experiments with simulated RSSIs...\n"
cd ScriptsForAllCases/1619_InPocket_crowded/
python realOfficeSimulationGeneral_print1619_8th_sec.py 1619_groundtruth_choose_8_to_57_inpocket_crowded.txt # creates 1619_mu.txt and creates/appends to numberOfNoSignals.txt
python calculate_mean_std_realworld.py 1619_groundtruth_choose_8_to_57_inpocket_crowded.txt 1619_mu.txt
python calculate_mean_std_speed_from_groundtruth.py 1619_groundtruth_choose_8_to_57_inpocket_crowded.txt
cat numberOfNoSignals.txt > ../../numberOfNoSignals.txt 

echo -e "Running In-Hand, Crowded experiments with simulated RSSIs...\n"
cd -
cd ScriptsForAllCases/1707_InHand_Crowded/
python realOfficeSimulationGeneral_print1707_150th_sec.py 1707_groundtruth_choose_1_to_50_inhand_crowded.txt # creates 1707_mu.txt and appends to numberOfNoSignals.txt
python calculate_mean_std_realworld.py 1707_groundtruth_choose_1_to_50_inhand_crowded.txt 1707_mu.txt
python calculate_mean_std_speed_from_groundtruth.py 1707_groundtruth_choose_1_to_50_inhand_crowded.txt
cat numberOfNoSignals.txt >> ../../numberOfNoSignals.txt 

echo -e "Running In-Pocket, Non-Crowded experiments with simulated RSSIs...\n"
cd -
cd ScriptsForAllCases/1841_InPocket_NonCrowded/
python realOfficeSimulationGeneral_print1841_10th_sec.py 1841_groundtruth_choose_5_to_54_inpocket_noncrowded.txt # creates 1841_mu.txt and appends to numberOfNoSignals.txt
python calculate_mean_std_realworld.py 1841_groundtruth_choose_5_to_54_inpocket_noncrowded.txt 1841_mu.txt
python calculate_mean_std_speed_from_groundtruth.py 1841_groundtruth_choose_5_to_54_inpocket_noncrowded.txt
cat numberOfNoSignals.txt >> ../../numberOfNoSignals.txt 

cd -
cp ScriptsForAllCases/1619_InPocket_crowded/1619_mu.txt  .
cp ScriptsForAllCases/1707_InHand_Crowded/1707_mu.txt .
cp ScriptsForAllCases/1841_InPocket_NonCrowded/1841_mu.txt .

echo -e "Calculating mean, standard deviation and cumulative distribution function for all cases...\n"
python calculate_mean_std_CDF_simulation_allCasesCombined.py 1619_groundtruth_choose_8_to_57_inpocket_crowded.txt 1619_mu.txt 1707_groundtruth_choose_1_to_50_inhand_crowded.txt 1707_mu.txt 1841_groundtruth_choose_5_to_54_inpocket_noncrowded.txt 1841_mu.txt 

echo -e "Plotting average error graph for each case...\n"
python avgErr_simulatedRSSIs.py realWorld_simulation_avgAcc.png

echo -e "Plotting #of no signal graph for each case...\n"
python noSignal_bar_simulatedRSSIs.py realworld_simulation_noSignal.png

echo -e "Combining average error and #of no signal plots into one image...\n"
convert +append realWorld_simulation_avgAcc.png realworld_simulation_noSignal.png realWorld_simulation_accuracies.png