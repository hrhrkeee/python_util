from IPython.display import HTML
import base64
import io
from pathlib import Path
import cv2
from typing import Any, Optional, Union

def convert_imageType_for_html(image: Any) -> Optional[Union[Path, str]]:
    """
    入力 image を HTML 表示用に変換する関数。

    - image が str または Path の場合:
        ファイルパスとして扱い、pathlib を用いて存在を確認。
        存在すれば Path オブジェクトを返し、存在しなければ None を返す。
    - image が Pillow の画像（PIL.Image.Image）の場合:
        PNG 形式でエンコードした data URL を返す。
    - image が OpenCV の画像（numpy.ndarray）の場合:
        cv2.imencode を用いて PNG 形式にエンコードした data URL を返す。
    - それ以外の場合は None を返す。
    """
    # 1. ファイルパスの場合（str, Path）
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.is_file() and path.exists():
            return path
        else:
            return None

    # 2. Pillow の画像の場合
    try:
        from PIL import Image
        if isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/png;base64,{img_base64}"
    except ImportError:
        pass

    # 3. OpenCV の画像の場合（通常 numpy.ndarray）
    try:
        import numpy as np
        if isinstance(image, np.ndarray):
            ret, buf = cv2.imencode('.png', image)
            if ret:
                img_bytes = buf.tobytes()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                return f"data:image/png;base64,{img_base64}"
            else:
                return None
    except ImportError:
        pass

    # 変換できない形式の場合
    return None

def display_images_grid_html(images, cols=3, row_height=200, labels=None, font_scale=1.0):
    """
    画像（または画像ファイルパス、または Pillow/OpenCV 画像オブジェクト）のリストを
    グリッド表示する HTML を生成する関数。

    引数:
      - images: 画像のリスト。各要素はファイルパス（str/Path）または Pillow/OpenCV 画像形式。
      - cols: グリッドの列数（デフォルトは 3）。
      - row_height: 各セル内で表示する画像の高さ（ピクセル）。アスペクト比は維持される。
      - labels: 画像下部に表示するラベルのリスト。指定されている場合、images と同数でなければ例外を発生させる。
      - font_scale: ラベル表示時のフォントサイズ倍率（デフォルトは 1.0）。
      
    画像およびラベルはセル中央に配置され、ラベル内の改行文字(\n)は <br> タグに変換されます。
    """
    # labels が指定されている場合、images と同じ数でなければエラー
    if labels is not None and len(labels) != len(images):
        raise ValueError("labels の要素数は images の要素数と一致する必要があります")
    
    # フォントサイズの設定（基本サイズ 14px に倍率をかける）
    base_font_size = 14
    font_size = base_font_size * font_scale

    html = "<table style='border-collapse: collapse;'>"
    for i, img in enumerate(images):
        if i % cols == 0:
            html += "<tr>"
        
        # 入力画像を HTML 表示用に変換
        converted = convert_imageType_for_html(img)
        if converted is None:
            converted = ""
        
        # 画像タグは height 属性で表示（アスペクト比は維持される）
        cell_content = f"<img src='{converted}' height='{row_height}px'>"
        
        # labels が指定されている場合、改行文字(\n)を <br> に置換して中央寄せで表示
        if labels is not None:
            label_text = labels[i].replace("\n", "<br>")
            cell_content += f"<br><div style='text-align: center; font-size: {font_size}px;'>{label_text}</div>"
        
        # セル全体を中央揃えにするため、text-align: center; を追加
        html += f"<td style='padding: 5px; text-align: center;'>{cell_content}</td>"
        
        if i % cols == cols - 1:
            html += "</tr>"
    if len(images) % cols:
        html += "</tr>"
    html += "</table>"
    return HTML(html)
