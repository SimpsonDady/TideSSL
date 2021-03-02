# ./run_experiment.sh self_jodie 0

model=$1
gpu=$2

date="$(date +'%Y%m%d')"
datasets=('lastfm' 'reddit' 'wikipedia')
groups=('10%' '5%' '1%' '0.5%')

dataset_idx=0
while [ $dataset_idx -le 2 ]
do
    dataset=${datasets[dataset_idx]}
    group_idx=0
    while [ $group_idx -le 3 ]
    do
        group=${groups[group_idx]}
        data_idx=7
        while [ $data_idx -le 10 ]
        do
            echo '=============' $dataset $group $data_idx '============='
            python3 $model.py --network $dataset$group'_'$data_idx --model jodie --epochs 50 --gpu $gpu >> Shell/$1$date.txt
            ./evaluate_all_epochs.sh $dataset$group'_'$data_idx interaction $gpu >> Shell/$1$date.txt
            ((data_idx+=1))
        done
        ((group_idx+=1))
    done
    ((dataset_idx+=1))
done

# dataset_idx=0
# while [ $dataset_idx -le 2 ]
# do
#     dataset=${datasets[dataset_idx]}
#     echo '=============' $dataset '============='
#     python3 $model.py --network $dataset --model jodie --epochs 50 --gpu $gpu >> Shell/$1$date.txt
#     ./evaluate_all_epochs.sh $dataset interaction $gpu >> Shell/$1$date.txt
#     ((dataset_idx+=1))
# done