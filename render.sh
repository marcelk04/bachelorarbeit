export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"

NUM_IMAGES_LIST=(
	128
	160
	256
)

for NUM_IMAGES in "${NUM_IMAGES_LIST[@]}"; do
	python src/data_generation/generate_images.py -s $SCENE_PATH -o output/marcus_$NUM_IMAGES --res 1024 --spp 128 -c $NUM_IMAGES
done