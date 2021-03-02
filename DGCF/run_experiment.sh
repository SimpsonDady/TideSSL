# ./run_experiment.sh self_DGCF 0

model=$1
gpu=$2

date="$(date +'%Y%m%d')"
datasets=('lastfm' 'reddit' 'wikipedia')
groups=('0.5%' '1%' '5%' '10%')
alphas=('0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9')

dataset_idx=0
while [ $dataset_idx -lt 3 ]
do
    dataset=${datasets[dataset_idx]}
    group_idx=0
    while [ $group_idx -lt 4 ]
    do
        group=${groups[group_idx]}
        data_idx=0
        while [ $data_idx -lt 11 ]
        do
            alpha_idx=2
            while [ $alpha_idx -lt 3 ]
            do
                alpha=${alphas[alpha_idx]}
                echo '=============' $dataset $group $data_idx $alpha '============='
#                 if [ $dataset_idx -eq 0 ] && [ $group_idx -le 2 ] || [ $data_idx -le 1 ]
#                 then
#                     ((alpha_idx+=1))
#                     continue
#                 fi
                python3 $model.py --network $dataset$group'_'$data_idx --model DGCF --epochs 100 --method attention --adj --alpha $alpha --gpu $gpu >> Shell/$1$date.txt
                ./evaluate_all_epochs.sh $dataset$group'_'$data_idx interaction $gpu $alpha >> Shell/$1$date.txt
                ((alpha_idx+=1))
            done
            ((data_idx+=1))
        done
        ((group_idx+=1))
    done
    ((dataset_idx+=1))
done


# dataset_idx=0
# while [ $dataset_idx -lt 3 ]
# do
#     dataset=${datasets[dataset_idx]}
#     echo '=============' $dataset '============='
#     python3 $model.py --network $dataset --model DGCF --epochs 100 --method attention --adj --gpu $gpu >> Shell/$1$date.txt
#     ./evaluate_all_epochs.sh $dataset interaction $gpu $alpha 0 99 >> Shell/$1$date.txt
#     ((dataset_idx+=1))
# done
