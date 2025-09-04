export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
IMAGE_PATH="output/marcus_64"
OUTPUT_PATH="output/fused"

SCENE_LIST=(
	unpolarized
	global
	direct
)

# Render dataset
# python src/data_generation/generate_images.py -s $SCENE_PATH -o $IMAGE_PATH --res 1024 --spp 128 -c 64

# Preprocessing (separate lighting, COLMAP)
# python src/preprocessing/separate_lighting.py -s $IMAGE_PATH
# python src/preprocessing/run_colmap.py -s $IMAGE_PATH -o $OUTPUT_PATH --include_test_cams

# Train unpolarized model
# python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/unpolarized/colmap -m $OUTPUT_PATH/unpolarized/model --disable_viewer --eval

# Train composite model
python submodules/gaussian-splatting/train_fused.py --source1 $OUTPUT_PATH/global/colmap --model $OUTPUT_PATH/composite/model --source2 $OUTPUT_PATH/direct/colmap --disable_viewer --eval

# Render test views and 360 videos 
python submodules/gaussian-splatting/render.py -m $OUTPUT_PATH/unpolarized/model --output $OUTPUT_PATH/results/unpolarized --skip_train
python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/unpolarized/model --output $OUTPUT_PATH/results/videos/unpolarized.mp4

python submodules/gaussian-splatting/render_separated.py -m $OUTPUT_PATH/composite/model --output $OUTPUT_PATH/results/global --skip_train --shs_idx=0 --source1 $OUTPUT_PATH/global/colmap --source2 $OUTPUT_PATH/direct/colmap
python submodules/gaussian-splatting/render360_separated.py -m $OUTPUT_PATH/composite/model --output $OUTPUT_PATH/results/videos/global.mp4 --shs_idx=0 --source1 $OUTPUT_PATH/global/colmap --source2 $OUTPUT_PATH/direct/colmap

python submodules/gaussian-splatting/render_separated.py -m $OUTPUT_PATH/composite/model --output $OUTPUT_PATH/results/direct --skip_train --shs_idx=1 --source1 $OUTPUT_PATH/global/colmap --source2 $OUTPUT_PATH/direct/colmap
python submodules/gaussian-splatting/render360_separated.py -m $OUTPUT_PATH/composite/model --output $OUTPUT_PATH/results/videos/direct.mp4 --shs_idx=1 --source1 $OUTPUT_PATH/global/colmap --source2 $OUTPUT_PATH/direct/colmap

# Reconstruct results from indirect/direct renders
python src/postprocessing/combine_images.py -g $OUTPUT_PATH/results/global/test/ours_30000 -d $OUTPUT_PATH/results/direct/test/ours_30000 -o $OUTPUT_PATH/results/composite/test/ours_30000
python src/postprocessing/combine_videos.py -g $OUTPUT_PATH/results/videos/global.mp4 -d $OUTPUT_PATH/results/videos/direct.mp4 -o $OUTPUT_PATH/results/videos/composite.mp4

# Evaluate metrics
for SCENE in "${SCENE_LIST[@]}"; do
	python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/$SCENE
done

python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/composite

# Plot results
python src/evaluation/plot_metrics.py -s $OUTPUT_PATH/results
python src/evaluation/plot_train_results.py -u $OUTPUT_PATH/unpolarized/model/train_results.json -c $OUTPUT_PATH/composite/model/train_results_combined.json -g $OUTPUT_PATH/composite/model/train_results1.json -d $OUTPUT_PATH/composite/model/train_results2.json -o $OUTPUT_PATH/results/graphs
