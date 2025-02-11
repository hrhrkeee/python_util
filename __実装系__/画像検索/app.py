import os
import subprocess
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import TkinterDnD, DND_FILES

from imageSearch.DatabaseManager import FAISSDatabaseManager
from imageSearch.ImageFeatureExtractor import ONNXImageFeatureExtractor

# TODO
# 1. データベースのパスを指定する
# 2. データベースに画像を登録する
# 3. 検索対象ディレクトリを指定して、インデックスに含まれていなかったら追加する


# 定数設定
QUERY_THUMB_SIZE = (180, 120)      # クエリー画像領域
RESULT_THUMB_SIZE = (150, 150)     # 基本サムネイルサイズ（100%時）
RESULT_THUMB_PADDING = 10          # 各セルの余白

def bind_cell_events(widget, cell):
    """セル内のウィジェットに、ホバー・クリック時のイベントをバインドする"""
    widget.bind("<Enter>", lambda e: cell.configure(cursor="hand2"))
    widget.bind("<Leave>", lambda e: cell.configure(cursor=""))
    widget.bind("<ButtonPress-1>", lambda e: cell.configure(bg="#dddddd"))
    widget.bind("<ButtonRelease-1>", lambda e: cell.configure(bg="white"))

class ImageSearchApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("画像検索")
        self.geometry("1000x700")
        self.minsize(450, 550)  # 最小横幅450px、縦550px
        
        # 初期設定
        self.query_image_path = None
        self.search_directory = None
        self.max_display_count = tk.IntVar(value=20)
        self.current_scale = 1.0  # 1.0 = 100%
        
        # 参照保持用
        self.thumbnail_images = []  # PhotoImageの参照保持（検索結果用）
        self.search_cells = []      # 各セル（Frame）を保持
        self.search_results = []      # 検索結果のPathオブジェクト
        
        self.create_menu()
        self.create_widgets()
    
    def create_menu(self):
        # メニューバーに「メニュー」→「初期状態に戻す」と「アプリを閉じる」を追加
        menu_bar = tk.Menu(self)
        menu = tk.Menu(menu_bar, tearoff=0)
        menu.add_command(label="初期状態に戻す", command=self.reset_app)
        menu.add_command(label="アプリを閉じる", command=self.quit)
        menu_bar.add_cascade(label="メニュー", menu=menu)
        self.config(menu=menu_bar)
    
    def reset_app(self):
        """アプリ起動時の状態に戻す"""
        self.query_image_path = None
        self.query_entry.delete(0, tk.END)
        self.draw_query_placeholder()
        
        self.search_directory = None
        self.dir_entry.delete(0, tk.END)
        
        self.max_display_count.set(20)
        self.current_scale = 1.0
        
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.thumbnail_images.clear()
        self.search_cells.clear()
    
    def create_widgets(self):
        default_bg = self.cget("bg")
        # ─ クエリー画像選択フレーム ─
        query_frame = ttk.LabelFrame(self, text="クエリー画像")
        query_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # 左側：サムネイル表示領域（Canvas）
        self.query_thumb_canvas = tk.Canvas(query_frame, width=QUERY_THUMB_SIZE[0],
                                             height=QUERY_THUMB_SIZE[1],
                                             bg=default_bg, highlightthickness=0)
        self.query_thumb_canvas.grid(row=0, column=0, padx=5, pady=5)
        self.draw_query_placeholder()
        self.query_thumb_canvas.drop_target_register(DND_FILES)
        self.query_thumb_canvas.dnd_bind("<<Drop>>", self.drop_query_image)
        
        # 右側：入力用 Entry と参照ボタン
        input_frame = ttk.Frame(query_frame)
        input_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.query_entry = ttk.Entry(input_frame)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        query_browse_btn = ttk.Button(input_frame, text="参照", command=self.browse_query_image)
        query_browse_btn.pack(side=tk.LEFT, padx=5)
        self.query_entry.drop_target_register(DND_FILES)
        self.query_entry.dnd_bind("<<Drop>>", self.drop_query_image)
        query_frame.columnconfigure(1, weight=1)
        
        # ─ 検索対象ディレクトリ選択フレーム ─
        dir_frame = ttk.LabelFrame(self, text="検索対象ディレクトリ")
        dir_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.dir_entry = ttk.Entry(dir_frame)
        self.dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        dir_browse_btn = ttk.Button(dir_frame, text="参照", command=self.browse_directory)
        dir_browse_btn.pack(side=tk.LEFT, padx=5)
        self.dir_entry.drop_target_register(DND_FILES)
        self.dir_entry.dnd_bind("<<Drop>>", self.drop_directory)
        
        # ─ 設定フレーム ─
        settings_frame = ttk.LabelFrame(self, text="設定")
        settings_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        max_label = ttk.Label(settings_frame, text="表示上限枚数:")
        max_label.pack(side=tk.LEFT, padx=5, pady=5)
        max_spin = ttk.Spinbox(settings_frame, from_=1, to=100,
                               textvariable=self.max_display_count, width=5)
        max_spin.pack(side=tk.LEFT, padx=5)
        
        # ─ 検索ボタン ─
        search_btn = ttk.Button(self, text="検索", command=self.search_images)
        search_btn.pack(side=tk.TOP, padx=10, pady=10)
        
        # ─ 検索結果表示エリア ─
        results_frame = ttk.LabelFrame(self, text="検索結果")
        results_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas = tk.Canvas(results_frame, bg=default_bg)
        self.scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.grid_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
    
    def draw_query_placeholder(self):
        """クエリー画像未指定時の領域を角丸点線枠と中央揃えテキストで表示"""
        self.query_thumb_canvas.delete("all")
        w, h = QUERY_THUMB_SIZE
        self._create_round_rect(self.query_thumb_canvas, 2, 2, w-2, h-2,
                                 radius=10, dash=(4,2), outline="gray")
        self.query_thumb_canvas.create_text(w/2, h/2, text="クエリー画像を選択してください",
                                             fill="gray", font=("Arial", 10), width=w-20, anchor="center")
    
    def _create_round_rect(self, canvas, x1, y1, x2, y2, radius=25, **kwargs):
        """Canvas に角丸矩形を描く補助関数（枠のみ）"""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, fill="", **kwargs)
    
    # ─ クエリー画像の処理 ─
    def browse_query_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            self.query_image_path = file_path
            self.query_entry.delete(0, tk.END)
            self.query_entry.insert(0, file_path)
            self.display_query_image()
    
    def drop_query_image(self, event):
        file_path = event.data
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        self.query_image_path = file_path
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, file_path)
        self.display_query_image()
    
    def display_query_image(self):
        if self.query_image_path and Path(self.query_image_path).exists():
            try:
                img = Image.open(self.query_image_path)
                img.thumbnail(QUERY_THUMB_SIZE)
                photo = ImageTk.PhotoImage(img)
                self.query_thumb_canvas.delete("all")
                self._create_round_rect(self.query_thumb_canvas, 2, 2, QUERY_THUMB_SIZE[0]-2, QUERY_THUMB_SIZE[1]-2,
                                         radius=10, dash=(4,2), outline="gray")
                self.query_thumb_canvas.create_image(QUERY_THUMB_SIZE[0]//2, QUERY_THUMB_SIZE[1]//2,
                                                       image=photo, anchor="center")
                self.query_thumb_canvas.image = photo
            except Exception as e:
                messagebox.showerror("エラー", f"クエリー画像の読み込みに失敗しました: {e}")
        else:
            self.draw_query_placeholder()
    
    # ─ 検索対象ディレクトリの処理 ─
    def browse_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.search_directory = dir_path
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, dir_path)
    
    def drop_directory(self, event):
        dir_path = event.data
        if dir_path.startswith("{") and dir_path.endswith("}"):
            dir_path = dir_path[1:-1]
        self.search_directory = dir_path
        self.dir_entry.delete(0, tk.END)
        self.dir_entry.insert(0, dir_path)
        
        
        
    def search_images_from_DB(self, db_manager, extractor, query_image_path, k=5):
        # 特徴抽出
        query_feature = extractor.extract_feature(query_image_path)
        # FAISSによる検索
        distances, indices = db_manager.search(query_feature, k)
        results = []
        for d, idx in zip(distances[0], indices[0]):
            # インデックスが-1の場合は該当なしとする
            if idx != -1:
                file_path = db_manager.file_paths[idx]
                results.append((d, file_path))
        return results
    
    # ─ 画像検索と検索結果グリッド表示 ─
    def search_images(self):
        if not self.search_directory or not Path(self.search_directory).is_dir():
            messagebox.showerror("エラー", "検索対象ディレクトリが正しく指定されていません。")
            return
        
        try:
            max_count = int(self.max_display_count.get())
        except:
            max_count = 20
                
        db_manager = FAISSDatabaseManager(index_file=Path("./localDB/FAISS/sampleDB/sampleDB.index"))
        extractor = ONNXImageFeatureExtractor(onnx_path="./model/ONNX/mobilenet_v2.onnx")
        results = self.search_images_from_DB(db_manager, extractor, self.query_image_path, k=max_count)
        # self.search_files = [str(r[1]) for r in results]
        self.search_results = [{"file_path":r[1], "distance":r[0]} for r in results]
        
        self.display_search_results(self.search_results)
    
    def display_search_results(self, search_results):
        # 検索結果エリアのクリア
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.thumbnail_images.clear()
        self.search_cells.clear()
        
        for result in search_results:
            try:
                img = Image.open(result["file_path"])
                new_size = (int(RESULT_THUMB_SIZE[0] * self.current_scale),
                            int(RESULT_THUMB_SIZE[1] * self.current_scale))
                img.thumbnail(new_size)
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_images.append(photo)
                cell_frame = tk.Frame(self.grid_frame, relief="groove", borderwidth=1, bg="white")
                
                # セル本体、画像ラベル、ファイル名ラベルすべてに共通イベントをバインド
                for widget in (cell_frame,):
                    bind_cell_events(widget, cell_frame)
                    
                top_lbl = ttk.Label(cell_frame, text=f"{result['distance']:.05}", wraplength=new_size[0])
                top_lbl.pack(padx=2, pady=(2,0))
                
                img_lbl = ttk.Label(cell_frame, image=photo)
                img_lbl.pack(padx=2, pady=2)
                bind_cell_events(img_lbl, cell_frame)
                
                filename_lbl = ttk.Label(cell_frame, text=f"{Path(result['file_path']).name}", wraplength=new_size[0])
                filename_lbl.pack(padx=2, pady=(0,2))
                bind_cell_events(filename_lbl, cell_frame)
                
                # ダブルクリックでエクスプローラー起動
                for widget in (cell_frame, img_lbl, filename_lbl):
                    widget.bind("<Double-Button-1>", lambda e, fp=result["file_path"]: self.open_explorer(fp))
                self.search_cells.append(cell_frame)
            except Exception as e:
                print(f"画像読み込みエラー {result["file_path"]}: {e}")
        
        self.re_layout_results()
    
    def re_layout_results(self):
        # Canvas の横幅に合わせてカラム数を再計算しグリッド配置
        self.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        if canvas_width <= 0:
            canvas_width = 640
        cell_width = int(RESULT_THUMB_SIZE[0] * self.current_scale) + RESULT_THUMB_PADDING
        cols = max(1, canvas_width // cell_width)
        
        for cell in self.search_cells:
            cell.grid_forget()
        row = 0
        col = 0
        for cell in self.search_cells:
            cell.grid(row=row, column=col, padx=5, pady=5, sticky="n")
            col += 1
            if col >= cols:
                col = 0
                row += 1
    
    def open_explorer(self, file_path):
        try:
            subprocess.run(['explorer', f'/select,{str(file_path)}'])
        except Exception as e:
            messagebox.showerror("エラー", f"エクスプローラーの起動に失敗しました: {e}")
    
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        self.re_layout_results()
        self.update_zoom_control_position()
    
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def update_zoom_control_position(self):
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.canvas.coords("zoom_control", cw-10, ch-10)
    
    def increase_zoom(self):
        if self.current_scale < 3.0:
            self.current_scale = round(self.current_scale + 0.1, 1)
            self.zoom_label.config(text=f"{int(self.current_scale*100)}%")
            if self.search_results:
                self.display_search_results(self.search_results)
    
    def decrease_zoom(self):
        if self.current_scale > 0.5:
            self.current_scale = round(self.current_scale - 0.1, 1)
            self.zoom_label.config(text=f"{int(self.current_scale*100)}%")
            if self.search_results:
                self.display_search_results(self.search_results)

def bind_cell_events(widget, cell):
    """セル内ウィジェットにホバー・クリックのイベントをバインドする"""
    widget.bind("<Enter>", lambda e: cell.configure(cursor="hand2"))
    widget.bind("<Leave>", lambda e: cell.configure(cursor=""))
    widget.bind("<ButtonPress-1>", lambda e: cell.configure(bg="#dddddd"))
    widget.bind("<ButtonRelease-1>", lambda e: cell.configure(bg="white"))

if __name__ == "__main__":
    app = ImageSearchApp()
    # 拡大率変更用コントロールをCanvasの右下に配置
    app.zoom_frame = ttk.Frame(app.canvas)
    btn_minus = ttk.Button(app.zoom_frame, text="-", command=app.decrease_zoom)
    btn_minus.pack(side=tk.LEFT)
    app.zoom_label = ttk.Label(app.zoom_frame, text=f"{int(app.current_scale*100)}%")
    app.zoom_label.pack(side=tk.LEFT, padx=5)
    btn_plus = ttk.Button(app.zoom_frame, text="+", command=app.increase_zoom)
    btn_plus.pack(side=tk.LEFT)
    app.canvas.create_window(app.canvas.winfo_width()-10, app.canvas.winfo_height()-10,
                             window=app.zoom_frame, anchor="se", tags="zoom_control")
    app.mainloop()
