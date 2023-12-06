import os
import cv2
import argparse
import tensorflow as tf
import numpy as np

if tf.__version__ >= '2.0':
    tf = tf.compat.v1

class ImageMattingPipeline:
    def __init__(self, model_dir: str, input_name: str = 'input_image:0', output_name: str = 'output_png:0'):
        model_path = os.path.join(model_dir, 'tf_graph.pb')
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found at {}".format(model_path))
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        with self.graph.as_default():
            self._session = tf.Session(config=config)
            with tf.gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            self.output = self._session.graph.get_tensor_by_name(output_name)
            self.input_name = input_name

    def preprocess(self, input_image_path: str):
        if not os.path.exists(input_image_path):
            raise FileNotFoundError("Input image file not found at {}".format(input_image_path))
        img = cv2.imread(input_image_path)
        img = img.astype(float)
        return {'img': img}

    def forward(self, input, output_mask=False, alpha_threshold=128):
        with self.graph.as_default(), self._session.as_default():
            feed_dict = {self.input_name: input['img']}
            output_img = self._session.run(self.output, feed_dict=feed_dict)
            if output_mask:
                alpha_channel = output_img[:, :, 3] 
                mask = np.zeros(alpha_channel.shape, dtype=np.uint8)
                mask[alpha_channel >= alpha_threshold] = 255
                return {'mask': mask}
            else:
                return {'output_img': output_img}

    def postprocess(self, inputs):
        if 'mask' in inputs:
            return {'mask': inputs['mask']}
        else:
            return {'output_img': inputs['output_img']}

def apply_filters(mask: np.array, closing_kernel: tuple = (5, 5), opening_kernel: tuple = (5, 5), 
                  blur_kernel: tuple = (3, 3), bilateral_params: tuple = (9, 75, 75), 
                  min_area: int = 2000) -> np.array:
    mask = mask.astype(np.uint8)
    closing_element = np.ones(closing_kernel, np.uint8)
    opening_element = np.ones(opening_kernel, np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_element)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, opening_element)
    smoothed_mask = cv2.GaussianBlur(opened_mask, blur_kernel, 0)
    edge_smoothed_mask = cv2.bilateralFilter(smoothed_mask, *bilateral_params)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_smoothed_mask, connectivity=8)
    large_component_mask = np.zeros_like(edge_smoothed_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_component_mask[labels == i] = 255
    return large_component_mask

def run_pipeline(pipeline, input_image_path: str, output_image_path: str, output_mask=False, apply_morphology=True):
    preprocessed = pipeline.preprocess(input_image_path)
    output = pipeline.forward(preprocessed, output_mask=output_mask)
    result = pipeline.postprocess(output)
    if 'mask' in result:
        if apply_morphology:
            result['mask'] = apply_filters(result['mask'], (5,5), (5,5), (5,5), (9,75,75))
        cv2.imwrite(os.path.splitext(output_image_path)[0] + "_mask.jpg", result['mask'])
    else:
        output_img = result['output_img']
        if output_img.shape[2] == 4:
            output_img = (output_img * 255).astype(np.uint8) if output_img.dtype != np.uint8 else output_img
            cv2.imwrite(os.path.splitext(output_image_path)[0] + ".png", output_img)
        else:
            output_img = (output_img * 255).astype(np.uint8) if output_img.dtype != np.uint8 else output_img
            cv2.imwrite(output_image_path, output_img)

def main(input_path, output_path, output_mask, apply_morphology=True):
    supported_extensions = ['.jpg', '.png', '.jpeg', '.webp']
    pipeline = ImageMattingPipeline(model_dir='cv_unet_universal-matting')
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename in os.listdir(input_path):
            if os.path.splitext(filename)[1].lower() in supported_extensions:
                input_image_path = os.path.join(input_path, filename)
                output_image_path = os.path.join(output_path, filename)
                
                preprocessed = pipeline.preprocess(input_image_path)
                output = pipeline.forward(preprocessed, output_mask=output_mask)
                result = pipeline.postprocess(output)

                if apply_morphology and 'mask' in result:
                    result['mask'] = apply_filters(result['mask'])
                
                if 'mask' in result:
                    cv2.imwrite(os.path.splitext(output_image_path)[0] + "_mask.jpg", result['mask'])
                else:
                    output_img = result['output_img']
                    if output_img.shape[2] == 4:
                        output_img = (output_img * 255).astype(np.uint8) if output_img.dtype != np.uint8 else output_img
                        cv2.imwrite(os.path.splitext(output_image_path)[0] + ".png", output_img)
                    else:
                        output_img = (output_img * 255).astype(np.uint8) if output_img.dtype != np.uint8 else output_img
                        cv2.imwrite(output_image_path, output_img)
                print(f"\rInference completed successfully for {input_image_path}", end='', flush=True)

    elif os.path.isfile(input_path) and os.path.splitext(input_path)[1].lower() in supported_extensions:
        run_pipeline(pipeline, input_path, output_path, output_mask=output_mask)

    else:
        print("Input path is neither a file nor a directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Matting Pipeline")
    parser.add_argument("-i", "--input", help="Input path: could be a directory containing the images or a single image", required=True)
    parser.add_argument("-o", "--output", help="Output path: a directory for the processed images if input is a directory or the output file if input is a single image", required=True)
    parser.add_argument("-m", "--mask", help="If set, output the mask image instead of the processed image.", action="store_true")
    parser.add_argument("-mo", "--morphology", help="If set, enables morphological processing on the mask.", action="store_true")
    args = parser.parse_args()
    main(args.input, args.output, args.mask, apply_morphology=args.morphology)
