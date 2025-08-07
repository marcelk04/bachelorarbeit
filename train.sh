export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
OUTPUT_PATH="output/0807_hair_10"

SCENE_LIST=(
	unpolarized
	indirect
	direct
)

python src/synthetic_gaussians/generate_images.py -s $SCENE_PATH -o $OUTPUT_PATH --res 1024 --spp 128
python src/synthetic_gaussians/pre_3dgs.py -w $OUTPUT_PATH

for SCENE in "${SCENE_LIST[@]}"; do
	echo "gaussian reconstruction for: $SCENE"

	# Train model
	python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/$SCENE/colmap -m $OUTPUT_PATH/$SCENE/model --disable_viewer --eval

	# Render results (test views + 360 video)
	python submodules/gaussian-splatting/render.py -m $OUTPUT_PATH/$SCENE/model --output $OUTPUT_PATH/results/$SCENE --skip_train
	python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/$SCENE/model --output $OUTPUT_PATH/results/videos/$SCENE.mp4
done

python src/synthetic_gaussians/combine_videos.py -i $OUTPUT_PATH/indirect/model/test/ours_30000/video/output.mp4 -d $OUTPUT_PATH/direct/model/test/ours_30000/video/output.mp4 -o $OUTPUT_PATH/results/videos/combined.mp4