# Get input arguments
CONFIG=$1
BENCH=$2
FOLDER=$3
GPUs=$4


#Prepare Paths
ws='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'
rs='/home/aming/MICRO/mobilenetv2/imagenet_pytorch_training'

cd $rs
mkdir -p $FOLDER
mkdir -p $FOLDER/$CONFIG
cd $ws

STDOUT="$rs/${FOLDER}/${CONFIG}/${BENCH}_%A_%a.stdout"
STDERR="$rs/${FOLDER}/${CONFIG}/${BENCH}_%A_%a.stderror"


# initial
SLURM_JOB_NAME="${BENCH}@${CONFIG}@${FOLDER}"

localMachineName=jarvis2
machineName='beluga' # befor submitting!


#Automaticallt decide the cluster to run on
if [[ 5 < $(expr match $(hostname)  ${machineName} ) ]];
then
       echo "This machine is: $machineName: $(hostname)"
       sbatch --output=$STDOUT --error=$STDERR -J ${SLURM_JOB_NAME}  $ws/RunBench_beluga.slurm

elif [[ 4 < $(expr match $(hostname)  'cedar' )  ]]
then
       echo "This machine is: cedar: $(hostname)"
       cd /scratch;
       sbatch --output=$STDOUT --error=$STDERR -J ${SLURM_JOB_NAME}  $ws/RunBench_cedar.slurm
       cd $ws
else
       if [[ $(hostname) == $localMachineName ]]
       then
           echo "This machine is: $(hostname) - running w/ vip on ${GPUs} GPUs"
           sbatch --output=$STDOUT --error=$STDERR --time=70:00:00 -p vip -J ${SLURM_JOB_NAME}  $ws/RunBench_local_${GPUs}.slurm
       else
           echo "This machine is: $(hostname) - running on ${GPUs} GPUs"
           sbatch --output=$STDOUT --error=$STDERR --time=48:00:00 -J ${SLURM_JOB_NAME}  $ws/RunBench_local_${GPUs}.slurm
       fi

fi

