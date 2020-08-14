#----------------------------------------------------------

ws=${TRAIN_HOME} # working folder path (root folder)

LR=$1 #learning rate
LD=$2 #gamma
LS=$3 #step size
ID=$4 #custom optimizer argument1
TS=$5 #custom optimizer argument2
EP=$6 #training epochs: 100
BS=$7 #batch size: 64
WD=$8 #weigh decay: '4e-5'
MO=$9 #momentum: 0.9
DB=${10} #custom opt flag: True/False
ST=${11} #LR update: fixedStep2 
QI=${12} #custom optimizer parameter1
QS=${13} #custom optimizer parameter2
QF=${14} #custom optimizer parameter3
MD=${15} #model name: 'mobilenetv2'
IN=${16} #input path: '/datasets/IMAGENET-UNCROPPED'

CP=./checkpoints # checkpointing folder
IS=224 # input size in pixels
DA=pytorch #dali-gpu # dataloader backend
WR=8 #number of workers
CR=checkpoints/checkpoint.pth.tar # path of checkpoints
echo "Tuning for LR=$LR LD=$LD LS=$LS ID=$ID WD=$WD TS=$TS EP=$EP BS=$BS MO=$MO DB=$DB ST=$ST QI=$QI QS=$QS QF=$QF MD=$MD IN=$IN"
echo "My home path is: $(pwd)" 
#----------------------------------------------------------

python $ws/imagenet_train.py \
    -a $MD \
    -d $IN \
    --epochs $EP \
    --lr-decay $ST \
    --step $LS \
    --gamma=$LD \
    --lr $LR \
    --init-decay $ID \
    --wd $WD \
    -c $CP \
    --input-size $IS \
    --batch-size $BS \
    --momentum $MO \
    -j $WR \
    --data-backend $DA \
    --resume $CR

#----------------------------------------------------------

#    --early-term True
