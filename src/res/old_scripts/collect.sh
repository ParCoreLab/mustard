#!/bin/bash
#SBATCH --job-name=lu_collect
#SBATCH -p palamut-cuda
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --export=ALL

cd /truba/home/iturimbetov/
bash env_local.sh
cd -
printenv

helpFunction()
{
    echo ""
    echo "Usage: $0 -a parameterA -b parameterB -c parameterC"
    echo -e "\t-a Description of what is parameterA"
    echo -e "\t-b Description of what is parameterB"
    echo -e "\t-c Description of what is parameterC"
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

# Print helpFunction in case parameters are empty
if [ -z "$runs" ]
then
    runs=10
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

STARPU_BUILD_DIR="/truba/home/iturimbetov/starpu/build"
MG_SAMPLES_DIR="../baselines/lu/MgPotrf"
MCUDAGRAPH_DIR="../mustard"

tile_sizes=(4 5 6)
matrix_sizes=(24000 36000 48000 72000 96000)
# T=6
# N=24000
TS=12
verb=""
for N in "${matrix_sizes[@]}"
do 
    mkdir -p logs/$N
    if [ $method -eq -1 ] || [ $method -eq 1 ]; then $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg -N=$N $verb -run=$runs >> logs/$N/log0_$N.log ; fi        
    for T in "${tile_sizes[@]}"
    do 
        if [ $gpu_count -eq -1 ]
        then 
            if [ $method -eq -1 ] || [ $method -eq 2 ]; then $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg -N=$N -T=$T -tiled $verb -workspace=256 -run=$runs >> logs/$N/log1_$N\_$T.log ; fi
                
            export CUDA_VISIBLE_DEVICES="0"
            for ((g = 1 ; g <= 8 ; g++ )); do
                echo "$CUDA_VISIBLE_DEVICES"
                if [ $method -eq -1 ] || [ $method -eq 3 ]; then $MPI_HOME/bin/mpirun -n $g $MCUDAGRAPH_DIR/lu_partg -N=$N -T=$T -subgraph $verb -workspace=256 -sm=26 -run=$runs >> logs/$N/log2_$N\_$T\_$g\GPU.log  ; fi
                # if [ $method -eq -1 ] || [ $method -eq 4 ]; then $MG_SAMPLES_DIR/cusolver_MgGetrf_example -N=$N -B=100 -r=$runs >> logs/$N/log3_$N\_$T\_$g\GPU.log ; fi
                    
                # for ((i = 0 ; i < $runs ; i++ )); do
                #     if [ $method -eq -1 ] || [ $method -eq 5 ]; then STARPU_SCHED=dmdas $STARPU_BUILD_DIR/examples/lu/lu_example_double -size $(($N)) -nblocks $TS >> logs/$N/log4_$N\_$TS\_$g\GPU.log ; fi
                #     #STARPU_SCHED=dmdas $STARPU_BUILD_DIR/examples/lu/lu_example_double -size $((24000)) -nblocks 12; done
                # done
                export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$g"
            done
        else 
            if [ $method -eq -1 ] || [ $method -eq 1 ]; then $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg -N=$N -T=$T $verb -workspace=256 -sm=20 -run=$runs ; fi
            if [ $method -eq -1 ] || [ $method -eq 2 ]; then $MPI_HOME/bin/mpirun -n 1 $MCUDAGRAPH_DIR/lu_partg -N=$N -T=$T -tiled $verb -workspace=256 -sm=20 -run=$runs ; fi
            if [ $method -eq -1 ] || [ $method -eq 3 ]; then $MPI_HOME/bin/mpirun -n $gpu_count $MCUDAGRAPH_DIR/lu_partg -N=$N -T=$T -subgraph $verb -workspace=256 -sm=20 -run=$runs ; fi
            
            export CUDA_VISIBLE_DEVICES="0"
            for ((g = 1 ; g < $gpu_count ; g++ )); do
                export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$g"
            done
            echo "$CUDA_VISIBLE_DEVICES"

            if [ $method -eq -1 ] || [ $method -eq 4 ]; then $MG_SAMPLES_DIR/cusolver_MgGetrf_example -N=$N -B=100 -r=$runs ; fi    
                
            for ((i = 0 ; i < $runs ; i++ )); do
                if [ $method -eq -1 ] || [ $method -eq 5 ]; then STARPU_SCHED=dmdas $STARPU_BUILD_DIR/examples/lu/lu_example_double -size $(($N)) -nblocks $TS ; fi
            done
        fi
    done
done
