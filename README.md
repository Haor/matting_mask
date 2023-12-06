# Image Matting Script Documentation

This project is an image matting/segmentation script implemented with TensorFlow and OpenCV, using the Damo Academy's BSHM generic matting model.

The script can perform matting on a single image or all the images in a directory, and can generate masks. Morphological processing can also be applied if needed.

## Dependencies
```shell
pip install -r requirements.txt
```
## Model Download
The model can be cloned from the following Git repository:
* https://www.modelscope.cn/damo/cv_unet_universal-matting

Navigate to the project directory and clone the repository:
```shell
cd your_project_directory
git clone https://www.modelscope.cn/damo/cv_unet_universal-matting.git
```

## Usage

### matting_mask
Run the script with the following command line parameters:
```shell
python matting_mask.py -i [input_path] -o [output_path] [-m] [-mo]
```

- `-i`/`--input`: Input path. Can be a single image file, or a directory containing multiple image files (supported image formats are: .jpg, .png, .jpeg, .webp).
- `-o`/`--output`: Output path. If the input is a directory, this should be a directory too. If the input is a single image file, this is the path to the output file.
- `-m`/`--mask`: If set, the script will output a mask image instead of the matting result.
- `-mo`/`--morphology`: If set, morphological operations will be enabled on the mask.
  
### split_video

Run the script with the following command line parameters:
```shell
python split_video.py -i [input_path] -o [output_directory] -f [fps] [-m] [-mo] [-a] [-t]
```

- `-i`/`--input`: The input path. This should be the path to the video file.
- `-o`/`--output`: The output directory. This should be a directory where the processed frames will be placed.
- `-f`/`--fps`: The framerate to process. This parameter determines how many frames are processed into images, default is 1.
- `-m`/`--mask`: If set, the script will output a mask image instead of the processed image.
- `-mo`/`--morphology`: If set, morphological operations will be enabled on the mask.
- `-a`/`--audio`: The output path for the audio from the video. If set, the script will extract the audio from the video and save it to the specified path.
- `-t`/`--trim`: The number of seconds to trim from the end of the video and audio. Default is 3 seconds, suitable for TikTok videos. If a specific value is not provided, the default value of 3 seconds will be used.

## Note
The model file should be stored in the 'cv_unet_universal-matting' directory and the filename should be 'tf_graph.pb'.
