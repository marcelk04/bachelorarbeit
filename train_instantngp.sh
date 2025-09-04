export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere_hair.xml"
IMAGE_PATH="output/marcus_160"
OUTPUT_PATH="output/instant_ngp"

SCENE_LIST=(
	unpolarized
	global
	direct
)

# Render dataset
python src/data_generation/generate_images.py -s $SCENE_PATH -o $IMAGE_PATH --res 1024 --spp 128 -c 160

# Preprocessing (separate lighting, COLMAP)
python src/preprocessing/separate_lighting.py -s $IMAGE_PATH
python src/preprocessing/run_colmap.py -s $IMAGE_PATH -o $OUTPUT_PATH

# Generate transforms.json
for SCENE in "${SCENE_LIST[@]}"; do
	python submodules/instant-ngp/scripts/colmap2nerf.py --images $OUTPUT_PATH/$SCENE/colmap/images --text $OUTPUT_PATH/$SCENE/colmap/manual --keep_colmap_coords --aabb_scale 16 --out $OUTPUT_PATH/$SCENE/transforms.json
done