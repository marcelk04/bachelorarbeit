export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
OUTPUT_PATH="output/gaussian_masks"

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
python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/unpolarized/colmap -m $OUTPUT_PATH/unpolarized/model --disable_viewer --eval

# Train composite model
python submodules/gaussian-splatting/train_tandem.py --source1 $OUTPUT_PATH/global/colmap --model1 $OUTPUT_PATH/global/model --source2 $OUTPUT_PATH/direct/colmap --model2 $OUTPUT_PATH/direct/model --disable_viewer --eval

# Render test views, 360 videos and evaluate metrics
for SCENE in "${SCENE_LIST[@]}"; do
	python submodules/gaussian-splatting/render.py -m $OUTPUT_PATH/$SCENE/model --output $OUTPUT_PATH/results/$SCENE --skip_train
	python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/$SCENE/model --output $OUTPUT_PATH/results/videos/$SCENE.mp4
	python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/$SCENE
done

# Reconstruct results from indirect/direct renders
python src/synthetic_gaussians/combine_videos.py -g $OUTPUT_PATH/results/videos/global.mp4 -d $OUTPUT_PATH/results/videos/direct.mp4 -o $OUTPUT_PATH/results/videos/composite.mp4
python src/synthetic_gaussians/combine_images.py -g $OUTPUT_PATH/results/global/test/ours_30000 -d $OUTPUT_PATH/results/direct/test/ours_30000 -o $OUTPUT_PATH/results/composite/test/ours_30000

# Evaluate metrics
python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/composite
python src/synthetic_gaussians/eval_results.py -s $OUTPUT_PATH
