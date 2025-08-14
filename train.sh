export CUDA_VISIBLE_DEVICES=7

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
OUTPUT_PATH="output/0812_train_tandem_2"

SCENE_LIST=(
	unpolarized
	indirect
	direct
)

# python src/synthetic_gaussians/generate_images.py -s $SCENE_PATH -o $OUTPUT_PATH --res 1024 --spp 128
# python src/synthetic_gaussians/pre_3dgs.py -w $OUTPUT_PATH

# Train unpolarized model
# python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/unpolarized/colmap -m $OUTPUT_PATH/unpolarized/model --disable_viewer --eval

# Train polarized model
python submodules/gaussian-splatting/train_tandem.py --source1 $OUTPUT_PATH/indirect/colmap --model1 $OUTPUT_PATH/indirect/model --source2 $OUTPUT_PATH/direct/colmap --model2 $OUTPUT_PATH/direct/model --disable_viewer --eval

# Render test views, 360 videos and evaluate metrics
for SCENE in "${SCENE_LIST[@]}"; do
	python submodules/gaussian-splatting/render.py -m $OUTPUT_PATH/$SCENE/model --output $OUTPUT_PATH/results/$SCENE --skip_train
	python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/$SCENE/model --output $OUTPUT_PATH/results/videos/$SCENE.mp4
	python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/$SCENE
done

# Reconstruct results from indirect/direct renders
python src/synthetic_gaussians/combine_videos.py -i $OUTPUT_PATH/results/videos/indirect.mp4 -d $OUTPUT_PATH/results/videos/direct.mp4 -o $OUTPUT_PATH/results/videos/combined.mp4
python src/synthetic_gaussians/combine_images.py -i $OUTPUT_PATH/results/indirect/test/ours_30000 -d $OUTPUT_PATH/results/direct/test/ours_30000 -o $OUTPUT_PATH/results/combined/test/ours_30000

# Evaluate metrics
python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/combined

