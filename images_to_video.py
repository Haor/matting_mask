import os
import cv2
import imageio.v2 as iio
import argparse
from tqdm import tqdm

def images_to_video(image_folder, video_name, fps):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = [img for img in os.listdir(image_folder) if os.path.splitext(img)[1] in supported_extensions]
    images.sort()

    if not images:
        print(f"No images found in the folder {image_folder}")
        return

    try:
        frame = iio.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

        for i, image in enumerate(tqdm(images, desc="Processing images")):
            video.write(cv2.cvtColor(iio.imread(os.path.join(image_folder, image)), cv2.COLOR_RGB2BGR))

        video.release()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a video from a sequence of images.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input images directory')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output video file')
    parser.add_argument('-f', '--fps', type=int, default=24, help='Frames per second')

    args = parser.parse_args()

    images_to_video(args.folder, args.output, args.fps)
    
    
    
#python images_to_video.py -i dance -o dance.mp4 -f 7
