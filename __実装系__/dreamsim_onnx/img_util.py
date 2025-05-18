import io
import cv2
import base64
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from IPython.display import HTML, display
from typing import Any, List, Optional, Union

def display_images_grid_html(
    images: Union[str, Path, np.ndarray, Image.Image, List[Union[str, Path, np.ndarray, Image.Image]]],
    labels: Optional[Union[str, List[str]]] = None,
    cols: int = 3,
    row_height: int = 200,
    font_scale: float = 1.0,
    save_html: Optional[str] = None
) -> None:
    def _convert_imageType_for_html(image: Any) -> Optional[Union[Path, str]]:
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                try:
                    resp = requests.get(image)
                    if resp.status_code == 200:
                        content_type = resp.headers.get("Content-Type", "image/png")
                        img_base64: str = base64.b64encode(resp.content).decode("utf-8")
                        return f"data:{content_type};base64,{img_base64}"
                    return None
                except Exception:
                    return None
            else:
                path: Path = Path(image)
                return path if path.is_file() and path.exists() else None
            
        if isinstance(image, (str, Path)):
            path: Path = Path(image)
            return path if path.is_file() and path.exists() else None
        if isinstance(image, Image.Image):
            buffered: io.BytesIO = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64: str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"
        if isinstance(image, np.ndarray):
            ret, buf = cv2.imencode('.png', image)
            if ret:
                img_base64: str = base64.b64encode(buf.tobytes()).decode("utf-8")
                return f"data:image/png;base64,{img_base64}"
            return None
        return None

    if not isinstance(images, (list, tuple)):
        images = [images]
    if labels is not None:
        if not isinstance(labels, (list, tuple)):
            labels = [labels] * len(images)
        elif len(labels) != len(images):
            raise ValueError("labels の要素数は images の要素数と一致する必要があります")
    
    base_font_size: int = 14
    font_size: float = base_font_size * font_scale

    html: str = "<table style='border-collapse: collapse;'>"
    for i, img in enumerate(images):
        if i % cols == 0:
            html += "<tr>"
        converted: Optional[Union[Path, str]] = _convert_imageType_for_html(img)
        if converted is None:
            converted = ""
        cell_content: str = f"<img src='{converted}' style='max-width: 100%; height: {row_height}px;'>"
        if labels is not None:
            label_text: str = labels[i].replace("\n", "<br>")
            cell_content += f"<br><div style='text-align: center; font-size: {font_size}px;'>{label_text}</div>"
        html += f"<td style='padding: 5px; text-align: center;'>{cell_content}</td>"
        if i % cols == cols - 1:
            html += "</tr>"
    if len(images) % cols:
        html += "</tr>"
    html += "</table>"

    display(HTML(html))
    
    if save_html is not None:
        html_save: str = f"<div style='max-width: 100%; overflow-x: auto;'>{html}</div>"
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(html_save)
