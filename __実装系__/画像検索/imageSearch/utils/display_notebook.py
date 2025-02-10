from IPython.display import HTML

def display_images_grid_thml(image_paths, cols=3, width=200):
    html = "<table style='border-collapse: collapse;'>"
    for i, path in enumerate(image_paths):
        if i % cols == 0:
            html += "<tr>"
        html += f"<td style='padding: 5px;'><img src='{path}' width='{width}px'></td>"
        if i % cols == cols - 1:
            html += "</tr>"
    if len(image_paths) % cols:
        html += "</tr>"
    html += "</table>"
    return HTML(html)
