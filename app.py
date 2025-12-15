import os
import re
import json
import time
import unicodedata
import gc
from io import BytesIO
from typing import Iterable, Tuple, Optional, List, Dict, Any

import gradio as gr
import numpy as np
import torch
import spaces
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import snapshot_download

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

print("🔄 Downloading Gliese-CUA-Tool-Call-8B model...")
local_dir = "./model/Gliese-CUA-Tool-Call-8B"
snapshot_download(
    repo_id="prithivMLmods/Gliese-CUA-Tool-Call-8B",
    local_dir=local_dir,
    resume_download=True,
    allow_patterns="Localization-8B/**",
)
model_path = os.path.join(local_dir, "Localization-8B")
print("✅ Model downloaded.")


print("🔄 Loading Gliese-CUA-Tool-Call-8B...")
try:
    processor_x = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model_x = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
except Exception as e:
    print(f"Failed to Gliese-CUA-Tool-Call-8B model: {e}")
    model_x = None
    processor_x = None

print("✅ Models loading sequence complete.")

def array_to_image(image_array: np.ndarray) -> Image.Image:
    if image_array is None: raise ValueError("No image provided.")
    return Image.fromarray(np.uint8(image_array))

def get_image_proc_params(processor) -> Dict[str, int]:
    ip = getattr(processor, "image_processor", None)
    
    default_min = 256 * 256
    default_max = 1280 * 1280

    patch_size = getattr(ip, "patch_size", 14)
    merge_size = getattr(ip, "merge_size", 2)
    min_pixels = getattr(ip, "min_pixels", default_min)
    max_pixels = getattr(ip, "max_pixels", default_max)

    size_config = getattr(ip, "size", {})
    if isinstance(size_config, dict):
        if "shortest_edge" in size_config:
            min_pixels = size_config["shortest_edge"]
        if "longest_edge" in size_config:
            max_pixels = size_config["longest_edge"]

    if min_pixels is None: min_pixels = default_min
    if max_pixels is None: max_pixels = default_max

    return {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

def apply_chat_template_compat(processor, messages: List[Dict[str, Any]], thinking: bool = True) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking=thinking)
        except TypeError:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    raise AttributeError("Could not apply chat template.")

def trim_generated(generated_ids, inputs):
    in_ids = getattr(inputs, "input_ids", None)
    if in_ids is None and isinstance(inputs, dict):
        in_ids = inputs.get("input_ids", None)
    if in_ids is None:
        return generated_ids
    return [out_ids[len(in_seq):] for in_seq, out_ids in zip(in_ids, generated_ids)]

def get_localization_prompt(task, image):
    guidelines = (
        "Localize an element on the GUI image according to my instructions and "
        "output a click position as Click(x, y) with x num pixels from the left edge "
        "and y num pixels from the top edge."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{guidelines}\n{task}"}
            ],
        }
    ]

def parse_click_response(text: str) -> List[Dict]:
    actions = []
    text = text.strip()
    
    matches_click = re.findall(r"(?:click|left_click|right_click|double_click)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text, re.IGNORECASE)
    for m in matches_click:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    matches_point = re.findall(r"point=\[\s*(\d+)\s*,\s*(\d+)\s*\]", text, re.IGNORECASE)
    for m in matches_point:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    matches_box = re.findall(r"start_box=['\"]?\(\s*(\d+)\s*,\s*(\d+)\s*\)['\"]?", text, re.IGNORECASE)
    for m in matches_box:
        actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})
    
    if not actions:
        matches_tuple = re.findall(r"(?:^|\s)\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:$|\s|,)", text)
        for m in matches_tuple:
            actions.append({"type": "click", "x": int(m[0]), "y": int(m[1]), "text": "", "norm": False})

    return actions

def create_localized_image(original_image: Image.Image, actions: list[dict]) -> Optional[Image.Image]:
    if not actions: return None
    img_copy = original_image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.load_default(size=18)
    except IOError:
        font = ImageFont.load_default()
    
    for act in actions:
        x = act['x']
        y = act['y']
        
        pixel_x, pixel_y = int(x), int(y)
            
        color = 'red' if 'click' in act['type'].lower() else 'blue'
        
        line_len = 15
        width = 4
        draw.line((pixel_x - line_len, pixel_y, pixel_x + line_len, pixel_y), fill=color, width=width)
        draw.line((pixel_x, pixel_y - line_len, pixel_x, pixel_y + line_len), fill=color, width=width)
        
        r = 20
        draw.ellipse([pixel_x - r, pixel_y - r, pixel_x + r, pixel_y + r], outline=color, width=3)
        
        label = f"{act['type']}"
        if act.get('text'): label += f": \"{act['text']}\""
        
        text_pos = (pixel_x + 25, pixel_y - 15)
        
        try:
            bbox = draw.textbbox(text_pos, label, font=font)
            padded_bbox = (bbox[0]-4, bbox[1]-2, bbox[2]+4, bbox[3]+2)
            draw.rectangle(padded_bbox, fill="yellow", outline=color)
            draw.text(text_pos, label, fill="black", font=font)
        except Exception as e:
            draw.text(text_pos, label, fill="white")

    return img_copy

@spaces.GPU
def process_screenshot(input_numpy_image: np.ndarray, task: str):
    if input_numpy_image is None: return "⚠️ Please upload an image.", None
    if not task.strip(): return "⚠️ Please provide a task instruction.", None

    input_pil_image = array_to_image(input_numpy_image)
    orig_w, orig_h = input_pil_image.size
    actions = []
    raw_response = ""

    if model_x is None: return "Error: UI-TARS model failed to load.", None
    print("Using UI-TARS Pipeline...")
    
    model, processor = model_x, processor_x
    ip_params = get_image_proc_params(processor)
    
    resized_h, resized_w = smart_resize(
        input_pil_image.height, input_pil_image.width,
        factor=ip_params["patch_size"] * ip_params["merge_size"],
        min_pixels=ip_params["min_pixels"], 
        max_pixels=ip_params["max_pixels"]
    )
    proc_image = input_pil_image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    
    messages = get_localization_prompt(task, proc_image)
    text_prompt = apply_chat_template_compat(processor, messages)
    
    inputs = processor(text=[text_prompt], images=[proc_image], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        
    generated_ids = trim_generated(generated_ids, inputs)
    raw_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    actions = parse_click_response(raw_response)
    
    if resized_w > 0 and resized_h > 0:
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h
        for a in actions:
            a['x'] = int(a['x'] * scale_x)
            a['y'] = int(a['y'] * scale_y)


    print(f"Raw Output: {raw_response}")
    print(f"Parsed Actions: {actions}")

    output_image = input_pil_image
    if actions:
        vis = create_localized_image(input_pil_image, actions)
        if vis: output_image = vis
            
    return raw_response, output_image

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.1em !important;}
"""
with gr.Blocks() as demo:
    gr.Markdown("# **Gliese-CUA-Tool-Call-8B-Localization 🖥️**", elem_id="main-title")

    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(label="Upload UI Image", type="numpy", height=500)
            
            task_input = gr.Textbox(
                label="Task Instruction",
                placeholder="e.g. Click on the search bar",
                lines=2
            )
            submit_btn = gr.Button("Call CUA Agent", variant="primary")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Visualized Action Points", elem_id="out_img", height=500)
            output_text = gr.Textbox(label="Agent Model Response", lines=10)

    submit_btn.click(
        fn=process_screenshot,
        inputs=[input_image, task_input],
        outputs=[output_text, output_image]
    )
    
if __name__ == "__main__":
    demo.queue(max_size=50).launch(theme=orange_red_theme, css=css, show_error=True)
