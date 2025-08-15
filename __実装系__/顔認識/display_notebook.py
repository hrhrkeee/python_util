import io
import cv2
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from IPython.display import HTML, display
from typing import Any, Optional, Union


def display_images_grid_html(images, labels=None, cols=3, row_height=200, font_scale=1.0):
    
    def _convert_imageType_for_html(image: Any) -> Optional[Union[Path, str]]:
        
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.is_file() and path.exists(): return path
            else: return None

        if isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"

        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ret, buf = cv2.imencode('.png', image_rgb)
            if ret:
                img_base64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                return f"data:image/png;base64,{img_base64}"
            else:
                return None

        return None
    
    if not isinstance(images, (list, tuple)):
        images = [images]
    if labels is not None and not isinstance(labels, (list, tuple)):
        labels = [labels] * len(images)
    
    if labels is not None and len(labels) != len(images):
        raise ValueError("labels の要素数は images の要素数と一致する必要があります")
    
    # フォントサイズの設定（基本サイズ 14px に倍率をかける）
    base_font_size = 14
    font_size = base_font_size * font_scale

    html = "<table style='border-collapse: collapse;'>"
    for i, img in enumerate(images):
        if i % cols == 0:
            html += "<tr>"
        
        converted = _convert_imageType_for_html(img)
        if converted is None:
            converted = ""
        
        cell_content = f"<img src='{converted}' height='{row_height}px'>"
        
        if labels is not None:
            label_text = labels[i].replace("\n", "<br>")
            cell_content += f"<br><div style='text-align: center; font-size: {font_size}px;'>{label_text}</div>"
        
        html += f"<td style='padding: 5px; text-align: center;'>{cell_content}</td>"
        
        if i % cols == cols - 1:
            html += "</tr>"
            
    if len(images) % cols:
        html += "</tr>"
        
    html += "</table>"
    return display(HTML(html))
