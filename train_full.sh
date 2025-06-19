export CUDA_VISIBLE_DEVICES=6

SCENE_PATH="scenes/marcus_light_sphere.xml"
OUTPUT_PATH="output/test_mask"

# python src/synthetic_gaussians/generate_images.py -s $SCENE_PATH -o $OUTPUT_PATH
python src/synthetic_gaussians/pre_3dgs.py -w $OUTPUT_PATH -s unpolarized

# python submodules/gaussian-splatting/train.py -s $OUTPUT_PATH/colmap -m $OUTPUT_PATH/model --disable_viewer
# python submodules/gaussian-splatting/render360.py -m $OUTPUT_PATH/model
