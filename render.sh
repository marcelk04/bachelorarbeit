export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"

NUM_IMAGES_LIST=(
	16
	32
	48
	64
	80
	128
	160
)

for NUM_IMAGES in "${NUM_IMAGES_LIST[@]}"; do
	# python src/data_generation/generate_images.py -s $SCENE_PATH -o output/marcus_${NUM_IMAGES}_white --res 1024 --spp 128 -c $NUM_IMAGES --white_background

	SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
	IMAGE_PATH="output/marcus_${NUM_IMAGES}_white"
	OUTPUT_PATH="output/nerf_${NUM_IMAGES}_white2"

	SCENE_LIST=(
		unpolarized
		global
		direct
	)

	# Render dataset
	# python src/data_generation/generate_images.py -s $SCENE_PATH -o $IMAGE_PATH --res 1024 --spp 128 -c 160

	# Preprocessing (separate lighting, COLMAP)
	python src/preprocessing/separate_lighting.py -s $IMAGE_PATH
	python src/preprocessing/copy_train_images.py -s $IMAGE_PATH -o $OUTPUT_PATH
	python src/data_generation/generate_transforms.py -s $SCENE_PATH -o $OUTPUT_PATH --res 1024 -c $NUM_IMAGES

	# Train and render
	for SCENE in "${SCENE_LIST[@]}"; do
		python submodules/instant-ngp/scripts/run.py $OUTPUT_PATH/$SCENE/transforms.json --save_snapshot $OUTPUT_PATH/$SCENE/model.ingp --screenshot_transforms $OUTPUT_PATH/$SCENE/transforms_test.json --screenshot_dir $OUTPUT_PATH/results/$SCENE/test/ours_30000/renders --width 1024 --height 1024 --n_steps 30000 --nerf_compatibility
	done

	python src/postprocessing/post_ingp.py -s $OUTPUT_PATH

	# Combine direct/global renders
	python src/postprocessing/combine_images.py -g $OUTPUT_PATH/results/global/test/ours_30000 -d $OUTPUT_PATH/results/direct/test/ours_30000 -o $OUTPUT_PATH/results/composite/test/ours_30000

	# Evaluate metrics
	for SCENE in "${SCENE_LIST[@]}"; do
		python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/$SCENE
	done

	python submodules/gaussian-splatting/metrics.py -m $OUTPUT_PATH/results/composite
	
done