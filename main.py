import argparse
import requests
import numpy as np
import gradio as gr
from PIL import Image

import logging
from logging import warning, info, error

from model import ModelInception, ModelInterpretation

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model compute: [cpu,cuda]')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for LIME model')

    return parser.parse_args()

def init_logging():
    logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.DEBUG)


def main():
    init_logging()
    cfg = get_config()

    # Models
    model_classification = ModelInception()
    model_interpretation = ModelInterpretation()

    def check_image_size(image, max_H=1000, max_W=1000):
        if (image.size[0] > max_W or image.size[1] > max_H):
            warning(f'Image exceeds size limits, compute is slow! MAX SIZE: {max_H} x {max_W}')
        if (image.size[0] > max_W*2 or image.size[1] > max_H*2):
            raise RuntimeError(f'Image more then exceeds size limits! MAX SIZE: {max_H} x {max_W}')


    def load_img_url(image_url):
        if 'svg' in image_url:
            warning('SVG is not supported.')
            return None, 0,0, 
    
        if len(image_url) > 0:
            info('Load image by URL.')
            image = Image.open(requests.get(image_url, stream=True).raw)
            check_image_size(image)
            return image, *image.size
        else:
            return None, 0,0, 

    def load_img(image):
        image = Image.fromarray(image)
        check_image_size(image)
        return image, *image.size

    def resize_image(image, height, width):
        info('Resize image.')
        if (height.isdigit() and width.isdigit()):
            image = Image.fromarray(image).resize((int(height), int(width)))
        return image

    def segment(image,height,width,n_superpixels,positive_chbx):
        info('LIME segmentation started')
        data = Image.fromarray(image)
        data = np.array(data)
        
        try:
            n_superpixels = int(n_superpixels)
            positive_chbx = bool(positive_chbx)
        except Exception as e: 
            error(e)

        try:
            model_interpretation(data, model_classification.batch_predict, cfg.batch_size)
        except Exception as e: 
            error(e)

        classes_ids = model_interpretation.get_classes()
        classes_names = [model_classification.idx2label[_id] for _id in classes_ids]
        
        bounds = model_interpretation.mark_boundaries(
            classes_ids[0], 
            positive_only=positive_chbx,
            num_features=n_superpixels)

        resize_image(image, height,width)

        model_classification.device(cfg.device)
        return bounds, gr.update(choices=classes_names, value=classes_names[0])

    def update_segment(value):
        # Update result segmentation
        _id = model_classification.label2idx[value]
        bounds = model_interpretation.mark_boundaries(_id)
        return bounds

    ###
    # UI
    ###
    with gr.Blocks() as UI:
        with gr.Row():
            image_url = gr.Textbox(label='Image URL', scale=2, 
                                   info='Please ender image URL')
            load_btn = gr.Button("Load", scale=1)

        with gr.Row():
            height = gr.Textbox(label="height")
            width  = gr.Textbox(label="width")
            resize_btn = gr.Button("Resize")

        with gr.Row():
            with gr.Column():
                n_superpixels = gr.Textbox(label="Number of superpixels", value=10)
                positive_chbx = gr.Checkbox(label='Positive labels only', value=False)
            segment_btn = gr.Button('Segmantation')    
        image = gr.Image()
        
        target_class_rd = gr.Radio(label="Target class", interactive=True, info='Select target class')

        ###
        # Callbacks
        ###
        load_btn.click(
            fn=load_img_url, 
            inputs=image_url, 
            outputs=[image, height, width], 
            api_name="load_img_url")
        
        resize_btn.click(
            fn=resize_image, 
            inputs=[image, height,width], 
            outputs=image, 
            api_name="resize")
        
        segment_btn.click(
            fn=segment, 
            inputs=[image,height,width,n_superpixels,positive_chbx], 
            outputs=[image, target_class_rd], 
            api_name="segmentation")
        
        target_class_rd.change(
            update_segment, 
            inputs=target_class_rd, 
            outputs=image)

        image.upload(
            load_img, 
            inputs= image, 
            outputs=[image, height, width], 
            api_name="load_img")

    if cfg.share:
        UI.launch(share=True)
    else:
        UI.launch(share=False)
    return 

if __name__ == '__main__':
    main()