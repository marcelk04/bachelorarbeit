export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
OUTPUT_PATH="output/fused_model"

SCENE_LIST=(
	unpolarized
	global
	direct
)

# Render dataset
# python src/synthetic_gaussians/generate_images.py -s $SCENE_PATH -o $OUTPUT_PATH --res 1024 --spp 128 -c 64

# Preprocessing (separate lighting, COLMAP)
# python src/synthetic_gaussians/pre_3dgs.py -w $OUTPUT_PATH

# Train unpolarized model
# python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/unpolarized/colmap -m $OUTPUT_PATH/unpolarized/model --disable_viewer --eval

# Train composite model
python submodules/gaussian-splatting/train_fused.py --source1 $OUTPUT_PATH/global/colmap --model $OUTPUT_PATH/composite/model --source2 $OUTPUT_PATH/direct/colmap --disable_viewer --eval

# Render test views, 360 videos and evaluate metrics
# python submodules/gaussian-splatting/render.py -m $OUTPUT_PATH/unpolarized/model --output $OUTPUT_PATH/results/unpolarized --skip_train
# python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/unpolarized/model --output $OUTPUT_PATH/results/videos/unpolarized.mp4

python submodules/gaussian-splatting/render_separated.py -m $OUTPUT_PATH/composite/model --output $OUTPUT_PATH/results/global --skip_train --shs_idx=0 --source1 $OUTPUT_PATH/global/colmap --source2 $OUTPUT_PATH/direct/colmap
# python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/global/model --output $OUTPUT_PATH/results/videos/global.mp4 --shs_idx=0

python submodules/gaussian-splatting/render_separated.py -m $OUTPUT_PATH/composite/model --output $OUTPUT_PATH/results/direct --skip_train --shs_idx=1 --source1 $OUTPUT_PATH/global/colmap --source2 $OUTPUT_PATH/direct/colmap
# python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/global/model --output $OUTPUT_PATH/results/videos/direct.mp4 --shs_idx=1

for SCENE in "${SCENE_LIST[@]}"; do
	python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/$SCENE
done

# Reconstruct results from indirect/direct renders
python src/synthetic_gaussians/combine_images.py -g $OUTPUT_PATH/results/global/test/ours_30000 -d $OUTPUT_PATH/results/direct/test/ours_30000 -o $OUTPUT_PATH/results/composite/test/ours_30000
# python src/synthetic_gaussians/combine_videos.py -g $OUTPUT_PATH/results/videos/global.mp4 -d $OUTPUT_PATH/results/videos/direct.mp4 -o $OUTPUT_PATH/results/videos/composite.mp4

# Evaluate metrics
python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/composite
python src/synthetic_gaussians/eval_results.py -s $OUTPUT_PATH
