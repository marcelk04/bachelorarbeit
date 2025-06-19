export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere.xml"
OUTPUT_PATH="output/test_3"

SCENE_LIST=(
	# unpolarized
	indirect
	direct
)

# python src/synthetic_gaussians/generate_images.py -s $SCENE_PATH -o $OUTPUT_PATH
python src/synthetic_gaussians/pre_3dgs.py -w $OUTPUT_PATH --skip_unpolarized

for SCENE in "${SCENE_LIST[@]}"; do
	echo "gaussian reconstruction for: $SCENE"

	python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/$SCENE/colmap -m $OUTPUT_PATH/$SCENE/model --disable_viewer
	python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/$SCENE/model
done

