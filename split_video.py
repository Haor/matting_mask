import os
import cv2
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
from matting_mask import ImageMattingPipeline, run_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def process_frame(frame_num, frame, pipeline, output_dir, output_mask, apply_morphology, processed_counter):
    try:
        frame_file = os.path.join(output_dir, f'{processed_counter:05d}.png')
        cv2.imwrite(frame_file, frame)
        run_pipeline(pipeline, frame_file, frame_file, output_mask=output_mask, apply_morphology=apply_morphology)
        if output_mask:
            os.remove(frame_file)
        return frame_file
    except Exception as e:
        logger.error(f"Error processing frame {frame_num}: {str(e)}")
        return None


def trim_last_seconds(video_clip, seconds):
    return video_clip.subclip(0, max(0, video_clip.duration - seconds))


def extract_audio_from_video(video_path, target_audio_path, trim_seconds):
    try:
        video_clip = VideoFileClip(video_path)
        if trim_seconds > 0:
            video_clip = trim_last_seconds(video_clip, trim_seconds)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(target_audio_path)
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path} to {target_audio_path}: {str(e)}")


def split_video(input_video, output_dir, fps, output_mask, apply_morphology, trim_seconds):
    pipeline = ImageMattingPipeline(model_dir='cv_unet_universal-matting')

    cap = cv2.VideoCapture(input_video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        logger.error(f'Cannot open video: {input_video}')
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f'Total frames: {total_frames}, video length: {total_frames / video_fps:.2f}s')

    processed_files = []
    processed_counter = 0
    trim_frames = 0
    if trim_seconds > 0:
        trim_frames = int(video_fps * trim_seconds)

    frame_num = 0
    with tqdm(total=total_frames, desc='Processing frames') as pbar:
        while frame_num < total_frames - trim_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % int(video_fps / fps) == 0:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future = executor.submit(process_frame, frame_num, frame, pipeline, output_dir, output_mask,
                                             apply_morphology, processed_counter)
                    frame_file = future.result()
                    if frame_file is not None:
                        processed_files.append(frame_file)
                    processed_counter += 1
                    pbar.set_postfix({'frame': frame_num}, refresh=True)
            pbar.update(1)
            frame_num += 1
    cap.release()
    logger.info(f'Total processed frames: {len(processed_files)}')


def process_directory(input_directory, output_directory, fps, output_mask, apply_morphology, trim_seconds, audio):
    if not os.path.isdir(input_directory):
        logger.error(f'Input directory does not exist: {input_directory}')
        return

    for filename in os.listdir(input_directory):
        input_path = os.path.join(input_directory, filename)
        if os.path.isfile(input_path) and filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_name = os.path.splitext(filename)[0]
            video_output_dir = os.path.join(output_directory, video_name)
            audio_path = audio if audio is not True else os.path.join(video_output_dir, "audio.mp3")

            split_video(input_path, video_output_dir, fps, output_mask, apply_morphology, trim_seconds)

            if audio is not None:
                extract_audio_from_video(input_path, audio_path, trim_seconds)

def process_single_video(input_path, output_directory, fps, output_mask, apply_morphology, trim_seconds, audio):
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    video_output_dir = os.path.join(output_directory, video_name)
    audio_path = audio if audio is not True else os.path.join(output_directory, video_name, "audio.mp3")

    split_video(input_path, video_output_dir, fps, output_mask, apply_morphology, trim_seconds)

    if audio is not None:
        extract_audio_from_video(input_path, audio_path, trim_seconds)

def process_input(args):
    trim_seconds = args.trim if args.trim is not None else 0

    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.fps, args.mask, args.morphology, trim_seconds, args.audio)
    else:
        process_single_video(args.input, args.output, args.fps, args.mask, args.morphology, trim_seconds, args.audio)        
        
def main():
    parser = argparse.ArgumentParser(description="Video Processing Pipeline")
    parser.add_argument("-i", "--input", help="Input video path or directory", required=True)
    parser.add_argument("-o", "--output", help="Output directory for the processed frames", required=True)
    parser.add_argument("-f", "--fps", help="Framerate to process", type=int, default=1)
    parser.add_argument("-m", "--mask", help="If set, output the mask image instead of the processed image.", action="store_true")
    parser.add_argument("-mo", "--morphology", help="If set, enables morphological processing on the mask.", action="store_true")
    parser.add_argument("-a", "--audio", help="Output path for the extracted audio", nargs='?', const=True, default=None)
    parser.add_argument("-t", "--trim", help="The number of seconds to trim from the end of the video and audio. Default is 3 seconds. For Tiktok video.", type=float, nargs='?', const=3, default=None)
    args = parser.parse_args()

    process_input(args)

if __name__ == "__main__":
    main()
