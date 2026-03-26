#!/bin/bash
# Auto-generated script for multiple views on GPU 7

VIEWS=(24)
NUM_RUNS=1
GPU_ID=5

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "Running experiments with views ${VIEWS[@]} (${NUM_RUNS} runs each) on GPU ${GPU_ID}"
echo "========================================="

# Loop through each view
for VIEW in "${VIEWS[@]}"; do
    echo "========================================="
    echo "Starting experiments with ${VIEW} views on GPU ${GPU_ID}"
    echo "========================================="
    
    # Loop through each run for this view
    for run in $(seq 1 $NUM_RUNS); do
        echo "[$(date)] Running experiment with ${VIEW} views - Run ${run}/${NUM_RUNS} on GPU ${GPU_ID}..."
        
        # Create output directory for this specific experiment
        OUTDIR="/data/akheirandish3/PaDIS_results/Dps_${VIEW}_NEW_OOD_default_large_full_intermediate_steps/run_${run}"
        mkdir -p $OUTDIR
        
        # Debug: Print the actual output directory being used
        #--network=/home/akheirandish3/PaDIS/network-snapshot-001602.pkl\
        echo "DEBUG: Created directory: $OUTDIR"
        echo "DEBUG: Directory contents before run:"
        ls -la "$OUTDIR" 2>/dev/null || echo "Directory is empty or doesn't exist"
        
        python3 dps_sampling_test.py \
            --network=/home/akheirandish3/PaDIS/training-runs_darker_4GPU_255/00000-aapm_3-uncond-ddpmpp-pedm-gpus4-batch24-fp32/network-snapshot-001345.pkl \
            --outdir="$OUTDIR" \
            --image_dir="/data2/akheirandish3/id_new_warped_images/ood_tumor_default_large/4" \
            --image_size=256 \
            --views="$VIEW" \
            --name=ct_parbeam \
            --steps=100 \
            --sigma_min=0.003 \
            --sigma_max=10 \
            --zeta=0.3 \
            --pad=24 \
            --num_runs=5 \
            --psize=56
        # Debug: Check what was actually created
        echo "DEBUG: Directory contents after run:"
        ls -la "$OUTDIR" 2>/dev/null || echo "Directory is empty"
        
        # Check if experiment was successful
        if [ $? -eq 0 ]; then
            echo "[$(date)] ✓ Successfully completed ${VIEW} views - Run ${run} on GPU ${GPU_ID}"
        else
            echo "[$(date)] ✗ Error in ${VIEW} views - Run ${run} on GPU ${GPU_ID}"
        fi
        
        echo "Results saved in: $OUTDIR"
        echo "----------------------------------------"
    done
    
    echo "[$(date)] Completed all ${NUM_RUNS} runs for ${VIEW} views on GPU ${GPU_ID}"
    echo "========================================="
done

echo "[$(date)] All experiments completed for views ${VIEWS[@]} on GPU ${GPU_ID}"
echo "========================================="
