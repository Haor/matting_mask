# Image Matting Script Documentation

This project is an image matting/segmentation script implemented with TensorFlow and OpenCV, using the Damo Academy's BSHM generic matting model (https://modelscope.cn/models/damo/cv_unet_universal-matting/).

The script can perform matting on a single image or all the images in a directory, and can generate masks. Morphological processing can also be applied if needed.

## Dependencies
This project requires the following libraries:
* TensorFlow
* OpenCV
* Numpy

## Model Download
The model can be cloned from the following Git repository:
* https://www.modelscope.cn/damo/cv_unet_universal-matting

Navigate to the project directory and clone the repository:
```shell
cd your_project_directory
git clone https://www.modelscope.cn/damo/cv_unet_universal-matting.git
```

## Usage

Run the script with the following command line parameters:
```shell
python matting_mask.py -i [input_path] -o [output_path] [-m] [-mo]
```

### Parameter Description
- `-i`/`--input`: Input path. Can be a single image file, or a directory containing multiple image files (supported image formats are: .jpg, .png, .jpeg, .webp).
- `-o`/`--output`: Output path. If the input is a directory, this should be a directory too. If the input is a single image file, this is the path to the output file.
- `-m`/`--mask`: If set, the script will output a mask image instead of the matting result.
- `-mo`/`--morphology`: If set, morphological operations will be enabled on the mask.

## Examples
Perform matting on all images in a directory, and save the results:
```shell
python matting_mask.py -i ./input_images -o ./output_images
```

Generate masks for all images in a directory, and save the results:
```shell
python matting_mask.py -i ./input_images -o ./output_images -m
```

Perform matting on a single image, and save the result:
```shell
python matting_mask.py -i ./input_images/image1.jpg -o ./output_images/image1_matting.png
```

Generate a mask for a single image, and save the result:
```shell
python matting_mask.py -i ./input_images/image1.jpg -o ./output_images/image1_mask.jpg -m
```

Perform matting on a single image, apply morphological operations, and save the result:
```shell
python matting_mask.py -i ./input_images/image1.jpg -o ./output_images/image1_matting.png -mo
```

Generate a mask for a single image, apply morphological operations, and save the result:
```shell
python matting_mask.py -i ./input_images/image1.jpg -o ./output_images/image1_mask.jpg -m -mo
```
## Note
The model file should be stored in the 'cv_unet_universal-matting' directory and the filename should be 'tf_graph.pb'. If the model file does not exist or is not found, the script will throw a FileNotFoundError.

Please make sure the input and output paths are correct, otherwise errors may occur.
