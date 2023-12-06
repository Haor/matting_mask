import os
import cv2
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
from matting_mask import ImageMattingPipeline, run_pipeline

def process_frame(frame_num, frame, pipeline, output_dir, output_mask, apply_morphology, processed_counter):
    frame_file = os.path.join(output_dir, f'{processed_counter:05d}.png')
    cv2.imwrite(frame_file, frame)
    run_pipeline(pipeline, frame_file, frame_file, output_mask=output_mask, apply_morphology=apply_morphology)
    if output_mask:
        os.remove(frame_file)
    return frame_file

def extract_audio_from_video(video_path, target_audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(target_audio_path)

def split_video(input_video, output_dir, fps, output_mask, apply_morphology):
    pipeline = ImageMattingPipeline(model_dir='cv_unet_universal-matting')

    cap = cv2.VideoCapture(input_video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print(f'Cannot open video: {input_video}')
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {total_frames}, video length: {total_frames / video_fps:.2f}s')

    processed_files = []
    processed_counter = 0

    frame_num = 0
    with tqdm(total=total_frames, desc='Processing frames') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % int(video_fps / fps) == 0:
                frame_file = process_frame(frame_num, frame, pipeline, output_dir, output_mask, apply_morphology, processed_counter)
                processed_files.append(frame_file)
                processed_counter += 1
                pbar.set_postfix({'frame': frame_num}, refresh=True)
            pbar.update(1)
            frame_num += 1

    cap.release()

    print(f'Total processed frames: {len(processed_files)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument("-i", "--input", help="Input video path", required=True)
    parser.add_argument("-o", "--output", help="Output directory for the processed frames", required=True)
    parser.add_argument("-f", "--fps", help="Frame rate to process", type=int, default=1)
    parser.add_argument("-m", "--mask", help="If set, output the mask image instead of the processed image.", action="store_true")
    parser.add_argument("-mo", "--morphology", help="If set, enables morphological processing on the mask.", action="store_true")
    parser.add_argument("-a", "--audio", help="Output path for the extracted audio", nargs='?', const=True, default="")
    args = parser.parse_args()

    audio_path = args.audio if args.audio and args.audio is not True else os.path.join(args.output, "audio.mp3")

    split_video(args.input, args.output, args.fps, args.mask, args.morphology)
    extract_audio_from_video(args.input, audio_path)
