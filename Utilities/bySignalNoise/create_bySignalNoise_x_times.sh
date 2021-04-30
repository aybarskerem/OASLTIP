#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"
NUMBER_OF_REPEATS=$1

takeMean() {
    local -n arr=$1
    sum=0
    for i in "${arr[@]}"
    do
        sum=$(bc <<< "$sum + $i")
    done
    echo $(bc <<< "scale=4 ; $sum / ${#arr[@]}")
}


takeStd() {
    local mean=$1
    local -n arr=$2
    sumOfSquares=0
    for i in "${arr[@]}"
    do
        sumOfSquares=$(bc <<< "$sumOfSquares + ($i-$mean)*($i-$mean)")
    done
    echo $(bc <<< "scale=4 ; sqrt($sumOfSquares / ${#arr[@]} )")
}



for signalNoise in 0 5 10 15 20 30
do
    fileName=$signalNoise"dBm_signalNoise"
    rm -f $fileName
    rm -f $fileName"_avgError.txt" 
    rm -f $fileName"_noSignalError.txt"  
done
rm -f means_avgAcc.txt
rm -f means_noSignalAcc.txt
rm -f stds_avgAcc.txt
rm -f stds_noSignalAcc.txt


for signalNoise in 0 5 10 15 20 30
do
    fileName=$signalNoise"dBm_signalNoise"
    for ((i=0; i<$NUMBER_OF_REPEATS; i++))
    do
        python bysignalNoise.py $signalNoise $i $fileName
    done

    avgErrorAcc=0
    noSignalErrorAcc=0
    avgErrorAcc_array=()
    while IFS= read -r line;
    do
        avgErrorAcc_array+=($line)
        #avgErrorAcc=$(bc <<< "scale=2 ; $avgErrorAcc + $line")
    done < $fileName"_avgError.txt" 

    
    mean_avgAcc=$(takeMean avgErrorAcc_array)
    echo "mean_avgAcc is for $fileName is: "$mean_avgAcc
    echo $mean_avgAcc >> means_avgAcc.txt

    standardDeviation_avgAcc=$(takeStd $mean_avgAcc avgErrorAcc_array)
    echo "standardDeviation_avgAcc is: " $standardDeviation_avgAcc
    echo $standardDeviation_avgAcc >> stds_avgAcc.txt

    noSignalErrorAcc_array=()
    while IFS= read -r line;
    do
        noSignalErrorAcc_array+=($line)
    done < $fileName"_noSignalError.txt" 


    mean_noSignalAcc=$(takeMean noSignalErrorAcc_array)
    echo "mean_noSignalAcc is for $fileName is: "$mean_noSignalAcc
    echo $mean_noSignalAcc >> means_noSignalAcc.txt

    standardDeviation_noSignalAcc=$(takeStd $mean_noSignalAcc noSignalErrorAcc_array)
    echo "standardDeviation_noSignalAcc is: " $standardDeviation_noSignalAcc
    echo $standardDeviation_noSignalAcc >> stds_noSignalAcc.txt

done


python avgErr_bar_signalNoise.py signalNoise_avgError.png
python noSignal_bar_signalNoise.py signalNoise_noSignal.png
convert +append signalNoise_avgError.png signalNoise_noSignal.png signalNoise_Accuracy.png