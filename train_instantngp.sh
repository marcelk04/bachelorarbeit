export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
IMAGE_PATH="output/marcus_160"
OUTPUT_PATH="output/instant_ngp_2"

SCENE_LIST=(
	unpolarized
	# global
	# direct
)

# Render dataset
# python src/data_generation/generate_images.py -s $SCENE_PATH -o $IMAGE_PATH --res 1024 --spp 128 -c 160

# Preprocessing (separate lighting, COLMAP)
python src/preprocessing/separate_lighting.py -s $IMAGE_PATH
python src/data_generation/generate_transforms.py -s $SCENE_PATH -o $IMAGE_PATH --res 1024 -c 160
python src/preprocessing/copy_train_images.py -s $IMAGE_PATH -o $OUTPUT_PATH
