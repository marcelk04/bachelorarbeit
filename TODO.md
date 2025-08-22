# TODOS

## Evaluate results

**Steps:**

- ~~render 360 videos~~
- ~~combine indirect + direct 360 video~~
- ~~render test cameras~~
- ~~combine indirect + direct images~~
- ~~evaluate metrics~~
- (generate nice graphs)
- ~~move results to separate folder~~

## Train polarized models together

- ~~test views can be evaluated on combined images~~
- ~~comparable to training of unpolarized model~~

**What we need:**

- ~~read in separate source paths~~
- ~~create two separate GaussianModels and Scenes~~
- ~~render independently~~
- ~~first evaluate loss separately~~
- ~~then add combined loss to both terms~~
- profit?

## Test perfomance

- less input images for polarized model
- lower model size for polarized model

## Output train results

- ~~log loss values for each iteration~~
- ~~evaluate perfomance regularly on test cams (PSNR, ...)~~

## Train NeRF model

## Further analyze generated gaussians

- mostly in hair area, e.g. rotation