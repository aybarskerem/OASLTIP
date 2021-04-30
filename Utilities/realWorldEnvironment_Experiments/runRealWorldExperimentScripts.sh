#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

echo -e "Copying GroundTruthPositions folder contents into RealWorldRSSIs folder...\n"
# RealWorldRSSIs required file copies
cp -a GroundTruthPositions/. RealWorldRSSIs/

echo -e "Copying GroundTruthPositions folder contents into corresponding cases in RealWorldRSSIs/ScriptsForAllCases folder...\n"
cp -a GroundTruthPositions/1619_groundtruth_choose_8_to_57_inpocket_crowded.txt RealWorldRSSIs/ScriptsForAllCases/1619_InPocket_crowded/
cp -a GroundTruthPositions/1707_groundtruth_choose_1_to_50_inhand_crowded.txt RealWorldRSSIs/ScriptsForAllCases/1707_InHand_Crowded/
cp -a GroundTruthPositions/1841_groundtruth_choose_5_to_54_inpocket_noncrowded.txt RealWorldRSSIs/ScriptsForAllCases/1841_InPocket_NonCrowded/

echo -e "Copying AllRSSIValues folder contents into corresponding cases in RealWorldRSSIs/ScriptsForAllCases folder...\n"
cp -a AllRSSIValues/AllValues1619_salt.csv RealWorldRSSIs/ScriptsForAllCases/1619_InPocket_crowded/
cp -a AllRSSIValues/AllValues1707_salt.csv RealWorldRSSIs/ScriptsForAllCases/1707_InHand_Crowded/
cp -a AllRSSIValues/AllValues1841_salt.csv RealWorldRSSIs/ScriptsForAllCases/1841_InPocket_NonCrowded/

echo -e "Copying GroundTruthPositions folder contents into SimulationGeneratedRSSIs folder...\n"
# SimulationGeneratedRSSIs required file copies
cp -a GroundTruthPositions/. SimulationGeneratedRSSIs/

echo -e "Copying GroundTruthPositions folder contents into corresponding cases in SimulationGeneratedRSSIs/ScriptsForAllCases folder...\n"
cp -a GroundTruthPositions/1619_groundtruth_choose_8_to_57_inpocket_crowded.txt SimulationGeneratedRSSIs/ScriptsForAllCases/1619_InPocket_crowded/
cp -a GroundTruthPositions/1707_groundtruth_choose_1_to_50_inhand_crowded.txt SimulationGeneratedRSSIs/ScriptsForAllCases/1707_InHand_Crowded/
cp -a GroundTruthPositions/1841_groundtruth_choose_5_to_54_inpocket_noncrowded.txt SimulationGeneratedRSSIs/ScriptsForAllCases/1841_InPocket_NonCrowded/

echo -e "Copying AllRSSIValues folder contents into corresponding cases in SimulationGeneratedRSSIs/ScriptsForAllCases folder...\n"
cp -a AllRSSIValues/AllValues1619_salt.csv SimulationGeneratedRSSIs/ScriptsForAllCases/1619_InPocket_crowded/
cp -a AllRSSIValues/AllValues1707_salt.csv SimulationGeneratedRSSIs/ScriptsForAllCases/1707_InHand_Crowded/
cp -a AllRSSIValues/AllValues1841_salt.csv SimulationGeneratedRSSIs/ScriptsForAllCases/1841_InPocket_NonCrowded/


echo "RealWorldRSSIs scripts are getting run..."
bash RealWorldRSSIs/createRealWorldRSSIs_figs.sh
echo "RealWorldRSSIs scripts have finished."

echo "SimulationGeneratedRSSIs scripts are getting run..."
bash SimulationGeneratedRSSIs/createSimulatedRSSIs_figs.sh
echo "SimulationGeneratedRSSIs scripts have finished."