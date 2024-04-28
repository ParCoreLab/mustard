#!/bin/bash
#SBATCH --job-name=lu_collect_8gpu
#SBATCH -p palamut-cuda
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --gres=gpu:8
#SBATCH --time=6:00:00
#SBATCH --export=ALL

# TODO
helpFunction()
{
    echo ""
    echo "Usage: $0 -r runs -g gpu_count -m method"
    echo -e "\t-r number of runs (11 by default)"
    echo -e "\t-g number of gpus (defaut is -1 tests for {1,2,4,8})"
    echo -e "\t-m method (defaut is -1 tests for all)"
    echo -e "\t\t method=1 is single-kernel cuSOLVER"
    echo -e "\t\t method=2 is single-GPU cudaGraph"
    echo -e "\t\t method=3 is mustard"
    echo -e "\t\t method=4 is multi-GPU cuSOLVER"
    echo -e "\t\t method=5 is starPU"
    echo -e "\t\t method=6 is slate"
    exit 1 # Exit script after printing help
}

while getopts "r:g:m:" opt
do
    case "$opt" in
        r ) runs="$OPTARG" ;;
        g ) gpu_count="$OPTARG" ;;
        m ) method="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
    esac
done

# cd /truba/home/iturimbetov/
# bash env_local.sh
# cd -
# printenv

# Print helpFunction in case parameters are empty
if [ -z "$runs" ]
then
    runs=11
fi
echo -n "$runs runs ";
if [ -z "$gpu_count" ]
then
    gpu_count=-1
    echo -n "on all gpus ";
else 
    echo -n "on $gpu_count gpus ";
fi
if [ -z "$method" ]
then
    method=-1
    echo "all methods ";
else 
    echo "method $method";
fi

BASE_SLATE_DIR="../baselines/slate"
BASE_STARPU_DIR="../baselines/starpu/lu"
MG_SAMPLES_DIR="../baselines/cusolver_Mg"
MCUDAGRAPH_DIR="../mustard"

tile_counts=(8)
# matrix_sizes=(1200)
matrix_sizes=(12000 24000 36000 48000 60000 72000)
# T=6
# N=24000
sm_count=100
workspace=2048
skip_gpu_regex='^(3|5|7)$'
TS=12
verb=""
timeoutcmd="timeout 15m "
for N in "${matrix_sizes[@]}"
do 
    outfolder=/truba_scratch/iturimbetov/mustard_logs/lu/$N
    mkdir -p $outfolder
    if [ $method -eq -1 ] || [ $method -eq 1 ]; then
        $timeoutcmd $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg \
            -N=$N $verb -run=$runs \
            &>> $outfolder/log0_$N.log ; 
    fi        
    for T in "${tile_counts[@]}"
    do 
        if [ $gpu_count -eq -1 ]
        then 
            if [ $method -eq -1 ] || [ $method -eq 2 ]; then
                $timeoutcmd $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg \
                    -N=$N -T=$T -tiled $verb -workspace=$workspace -sm=$sm_count -run=$runs \
                    &>> $outfolder/log1_$N\_$T.log ; 
            fi
                
            export CUDA_VISIBLE_DEVICES="0"
            for ((g = 1 ; g <= 8 ; g++ )); do
                echo "$CUDA_VISIBLE_DEVICES"
                if [[ $g =~ $skip_gpu_regex ]]; then
                    echo "skip"
                else
                    B=$(( N / T ))
                    echo $B
                    if [ $method -eq -1 ] || [ $method -eq 3 ]; then
                        $timeoutcmd $MPI_HOME/bin/mpirun -n $g $MCUDAGRAPH_DIR/lu_partg \
                            -N=$N -T=$T -subgraph $verb -workspace=$workspace -sm=$sm_count -run=$runs \
                            &>> $outfolder/log2_$N\_$T\_$g\GPU.log  ; 
                    fi
                    if [ $method -eq -1 ] || [ $method -eq 4 ]; then
                        $timeoutcmd $MG_SAMPLES_DIR/cusolver_MgGetrf_example \
                            -N=$N -B=$B -r=$runs \
                            &>> $outfolder/log3_$N\_$T\_$g\GPU.log ; 
                    fi
                        
                    for ((i = 0 ; i < $runs ; i++ )); do
                        if [ $method -eq -1 ] || [ $method -eq 5 ]; then 
                            STARPU_NCPU=0 STARPU_DISABLE_PINNING=1 STARPU_SCHED=dmdas \
                            STARPU_PERF_MODEL_HOMOGENEOUS_CUDA=1 STARPU_PERF_MODEL_HOMOGENEOUS_CPU=1 \
                            STARPU_WORKERS_COREID=1-9 \
                            $timeoutcmd $BASE_STARPU_DIR/lu_example_double -size $N -nblocks $T \
                            &>> $outfolder/log4_$N\_$T\_$g\GPU.log ; 
                        fi
                        #STARPU_SCHED=dmdas $STARPU_BUILD_DIR/examples/lu/lu_example_double -size $((24000)) -nblocks 12; done
                    done
                    if [ $method -eq -1 ] || [ $method -eq 6 ]; then 
                        $timeoutcmd $MPI_HOME/bin/mpirun -n $g $BASE_SLATE_DIR/lu_slate \
                        -n=$N -b=$B -runs=$runs &>> $outfolder/log5_$N\_$T\_$g\GPU.log ; 
                    fi
                fi
                export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$g"
            done
        else 
            export CUDA_VISIBLE_DEVICES="0"
            for ((g = 1 ; g < $gpu_count ; g++ )); do
                export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$g"
            done
            echo "$CUDA_VISIBLE_DEVICES"
            # if [ $method -eq -1 ] || [ $method -eq 1 ]; then $timeoutcmd $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg -N=$N $verb -run=$runs ; fi
            # if [ $method -eq -1 ] || [ $method -eq 2 ]; then $timeoutcmd $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg -N=$N -T=$T -tiled $verb -workspace=$workspace -sm=$sm_count -run=$runs ; fi
            if [ $method -eq -1 ] || [ $method -eq 3 ]; then
                $timeoutcmd $MPI_HOME/bin/mpirun -n $g $MCUDAGRAPH_DIR/lu_partg \
                    -N=$N -T=$T -subgraph $verb -workspace=$workspace -sm=$sm_count -run=$runs \
                    &>> $outfolder/log2_$N\_$T\_$g\GPU.log  ; 
            fi
            if [ $method -eq -1 ] || [ $method -eq 4 ]; then
                $timeoutcmd $MG_SAMPLES_DIR/cusolver_MgGetrf_example \
                    -N=$N -B=$B -r=$runs \
                    &>> $outfolder/log3_$N\_$T\_$g\GPU.log ; 
            fi
                
            for ((i = 0 ; i < $runs ; i++ )); do
                if [ $method -eq -1 ] || [ $method -eq 5 ]; then 
                    STARPU_NCPU=0 STARPU_DISABLE_PINNING=0 STARPU_SCHED=dmdas \
                    STARPU_PERF_MODEL_HOMOGENEOUS_CUDA=1 STARPU_PERF_MODEL_HOMOGENEOUS_CPU=1 \
                    $timeoutcmd $BASE_STARPU_DIR/lu_example_double -size $N -nblocks $T \
                    &>> $outfolder/log4_$N\_$T\_$g\GPU.log ; 
                fi
                #STARPU_SCHED=dmdas $STARPU_BUILD_DIR/examples/lu/lu_example_double -size $((24000)) -nblocks 12; done
            done
            if [ $method -eq -1 ] || [ $method -eq 6 ]; then 
                $timeoutcmd $MPI_HOME/bin/mpirun -n $g $BASE_SLATE_DIR/lu_slate \
                -n=$N -b=$B -runs=$runs &>> $outfolder/log5_$N\_$T\_$g\GPU.log ; 
            fi
        fi
    done
done
