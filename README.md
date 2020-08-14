# Neural Network Training Management Framework with PyTorch 


The training script is edited from reproduction of MobileNet V2 architecture as described in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


| Table of Contents |  |
|-------------------|--|
| Tuning            |  |
| Things to set     |  |
| Parameter counts  |  |
| Dataset           |  |
| Clusters          |  |
| File transfer     |  |



## Tuning
For training, introduce your model parameters in models/parameters.json and call it similar to examples in `runOnCluster.sh`

Hierarchy:

```
runOnCluster.sh [model_name, sub-conf, out_dir]
└── tune.sh > tuning.log
    ├── tuning.py [*model_name] > tuning.runs
    └── submit_tune.sh
        └── RunBench[_cedar/_beluga/_local].slurm
            └── run_tune.sh (inside Singularity)
                └── imagenet_train.py [generated_tuning_arguments]
```

The output of each training is at: `out_dir/_config_/_runName_jobArrayId.std[out/error]`
The log of each training is at: `out_dir/tmps/_run_name@_config_@_out-dir_/checkpoints/log.txt`

Each RunBench have diff groups, partition, scheduling time, singularity dir binding.
the submit_tune.sh script will automatically figure out which cluster you are using now by simple pattern matching on the hostname.

 + Local: RunBench_local_1/8.slurm: 1 or 8 GPUs on local clusters
 + Compute Canada: RunBench_beluga.slurm and RunBench_cedar.slurm. no vip partition
  
 

The job name that is passed to RunBench.slurm has 3 field separated by '@': e.g. `ImageNet_mobileNetv2@0.0625_0.25_2_0.9_2000000_10_64_4e-05_0_False_fixedStep@tuning`
config is passed like this: `0.0625_0.18_2_0.9_2000000_10_64_4e-05_0_True_fixedStep

e.g. output: `tuning/0.0625_0.18_2_0.9_2000000_10_64_4e-05_0_False_fixedStep/ImageNet_mobileNetv2_2201_1.stdout`
e.g. log: `tuning/tmps/ImageNet_mobileNetv2@0.0625_0.25_2_0.9_2000000_10_64_4e-05_0_False_fixedStep@tuning/checkpoints/log.txt`
### Helper functions

 + summary_all_models.sh
 + tuning_best.sh: reports best accuracy for group of training
 + tuning_worst.sh

NOTE: put this in your ~/.bashrc file: `alias squeueMe='nvidia-smi2;squeue -u your-username -o "%.18i %120j %20S %10L %.10M %.6D %.2t"'

### Things to set

in `model/parameters.json` ****:
 - add the tuning parameters in json(dictionary) format for each model and sub-configuration

in `submit_tune.sh` and `RunBench.slurm`:
 - ws: path for scripts. set ${TRAIN_HOME}
 - rs: path to dump outputs. set ${TRAIN_HOME}

in `RunBench_x.slurm`:
 - SING_IMG='.../custom.simg': which singularity image to use


## Parameters

Result of `bash summary_all_models.sh` (NOTE: missing efficientnet-bx capability):

|Model             | tested |parameter count  ||Model              | tested |parameter count  |
|------------------|--------|----------------:||-------------------|--------|----------------:|
|alexnet           |        |         61100840|| resnet18          |   X    |        11689512 |
|densenet121       |        |     7978856     || resnet34          |   X    |        21797672 |
|densenet161       |        |     28681000    || resnet50          |   X    |        25557032 |
|densenet169       |        |     14149480    || resnext101_32x8d  |        |        88791336 |
|densenet201       |        |     20013928    || resnext50_32x4d   |        |         25028904|
|mobilenetv2       |   X    |     3504872     || squeezenet1_0     |   X    |   1248424       |
|mobilenet_v2      |   X    |    3504872      || squeezenet1_1     |   X    |   1235496       |
|shufflenet_v2_x0_5|   X    |      1366792    || vgg11             |        |   132863336     |
|shufflenet_v2_x1_0|   X    |      2278604    || vgg11_bn          |        |        132868840|
|shufflenet_v2_x1_5|   X    |      3503624    || vgg13             |        |   133047848     |
|shufflenet_v2_x2_0|   X    |      7393996    || vgg13_bn          |        |        133053736|
|inception_v3      |        |    27161264     || vgg16             |   X    |   138357544     |
|mnasnet0_5        |        |      2218512    || vgg16_bn          |        |        138365992|
|mnasnet0_75       |        |     3170208     || vgg19             |   X    |   143667240     |
|mnasnet1_0        |        |      4383312    || vgg19_bn          |        |        143678248|
|mnasnet1_3        |        |      6282256    || wide_resnet101_2  |        |        126886696|
|resnet101         |        |       44549160  || wide_resnet50_2   |   X    |         68883240|
|resnet152         |        |       60192808  || googlenet         |        |       13004888  |
| efficientnet-b0  |   X    |       5288548   || efficientnet-b2   |   X    |        9109994  |



## Clusters

Compute Canada Beluga:

  - Run under your-accunt name (#SBATCH --account=your-account)
  - Up to 7 days
  - Min time limit of 1 hour per job
  - Max 1000 running jobs
  - /scratch have a max of #1000K files limitation ( moving IMAGENET to /project)
  - Files under /project should have the `group=your-group`
  - Cluster info: run `partition-stats` command
  - Priority stats: run `sshare -l` command
  - Need to load singularity (done in RunBench) if using the Singularity image (i.e. module load singularity)
  - Need to load correct version of python `module load python/3.7.4` (put it in ~/.bashrc)
  - Transfer large data: https://globus.computecanada.ca (https://docs.computecanada.ca/wiki/Globus)
  - Scheduling info: https://docs.computecanada.ca/wiki/Job_scheduling_policies
  - GPU info: https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm


## File transfer

You need to transfer files with `your-group` group so you don't exceed the quota on /projcts:

For example moving files from /scratch: `rsync --info=progress2 --chown=your-user:your-group -r path_to/IMAGENET-UNCROPPED destination_path/`

parallel data movement:

```

ls src/IMAGENET-UNCROPPED/train/ | xargs -n1 -P32 -I% rsync --info=progress2 --chown=your-user:your-group -r src/IMAGENET-UNCROPPED/train/% dst/IMAGENET-UNCROPPED/train/
ls src/IMAGENET-UNCROPPED/val/   | xargs -n1 -P32 -I% rsync --info=progress2 --chown=your-user:your-group -r src/MAGENET-UNCROPPED/val/%   dst/IMAGENET-UNCROPPED/val/

```

Run the following to test copy speed:
`bash test_transfer.sh`

The output will be :

```
/project/.../your-user --> /project/.../your-user:
test.tar.gz                                                                                                                                                        100%  145MB 205.7MB/s   00:00
/project/.../your-user --> /home/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 217.3MB/s   00:00
/project/.../your-user --> /scratch/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 220.6MB/s   00:00
/home/your-user --> /project/.../your-user:
test.tar.gz                                                                                                                                                        100%  145MB 219.7MB/s   00:00
/home/your-user --> /home/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 219.7MB/s   00:00
/home/your-user --> /scratch/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 220.0MB/s   00:00
/scratch/your-user --> /project/.../your-user:
test.tar.gz                                                                                                                                                        100%  145MB 216.1MB/s   00:00
/scratch/your-user --> /home/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 217.8MB/s   00:00
/scratch/your-user --> /scratch/your-user:
test.tar.gz                                                                                                                                                        100%  145MB 218.2MB/s   00:00
```

Run the following commands to test drive Wr/Rd speed:

`cd /drive_to_test/user/...`

Writing: 
`sync; dd if=/dev/zero of=tempfile bs=1M count=1024; sync`

output:

```
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.993727 s, 1.1 GB/s
```

Reading: 
`dd if=tempfile of=/dev/null bs=1M count=1024`

```
1024+0 records in
1024+0 records out
1073741824 bytes (1.1 GB, 1.0 GiB) copied, 0.251305 s, 4.3 GB/
```
