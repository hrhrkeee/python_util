import io, random
import torch
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import contextlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

class ImageDataset(Dataset):
    TARGET_EXT = ("jpg", "jpeg", "png")
    
    def __init__(self, image_input, transform=None, img_size=(512, 512), data_size=None, random_sample=True):
        """
        Args:
            image_input (str or list): ディレクトリのパス、画像パスのリスト、またはnumpy.ndarrayのリスト。
            transform (callable, optional): サンプルに適用するオプションの変換。
            data_size (int, optional): データセットのサイズを指定する。
            random_sample (bool, optional): Trueならランダムにdata_size分の画像を選択。
        """
        
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        self.data_size = data_size
        self.random_sample = random_sample


        if isinstance(image_input, str) or isinstance(image_input, Path):
            self.image_paths = [str(path) for path in Path(image_input).glob("*") if path.suffix[1:] in self.TARGET_EXT]
        elif isinstance(image_input, list):
            if all(isinstance(i, np.ndarray) for i in image_input):
                self.images = image_input
            else:
                self.image_paths = image_input
        else:
            raise TypeError("image_input must be a directory path, a list of image paths, or a list of numpy.ndarrays.")
        
        # data_sizeが指定されているかつ、元のデータ数より多い場合はエラーを出す
        if self.data_size is not None and self.data_size > self.__len__():
            raise ValueError(f"data_size {self.data_size} is greater than the number of images in the dataset {self.__len__()}.")

        # random_sampleがTrueの場合、ランダムにdata_size分の画像を選択
        if self.random_sample and self.data_size is not None:
            if hasattr(self, 'image_paths'):
                self.image_paths = random.sample(self.image_paths, self.data_size)
            elif hasattr(self, 'images'):
                self.images = random.sample(self.images, self.data_size)
                

    def __len__(self):
        # 画像パスのリストか、numpy.ndarrayのリストの長さを返す
        if hasattr(self, 'image_paths'):
            return len(self.image_paths)
        elif hasattr(self, 'images'):
            return len(self.images)

    def __getitem__(self, idx) -> torch.Tensor:
        # 画像をnumpy.ndarrayとしてロードする
        if hasattr(self, 'image_paths'):
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
        elif hasattr(self, 'images'):
            image = self.images[idx]
            # numpy.ndarrayをPILイメージに変換
            image = Image.fromarray(image)
        
        # transformを適用する
        image = self.transform(image)

        return image
    
    def get_all_tensor(self):
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])
        
        return torch.stack([self[i] for i in range(len(self))])


class TensorPlotter:
    def __init__(self, tensors, device="cpu", labels=None):
        self.tensors = tensors
        self.num_tensors = len(tensors)
        self.labels = labels or [f"Tensor {i+1}({len(self.tensors[i])})" for i in range(self.num_tensors)]
        assert self.num_tensors == len(self.labels), "Number of tensors and labels should match."
        
        # self.color_map =  plt.cm.Set1([i for i in range(self.num_tensors)])
        # self.color_map =  plt.cm.Dark2([i for i in range(self.num_tensors)])
        self.cmap =  np.array(plt.get_cmap("tab10").colors)
        # self.color_map =  plt.cm.hsv(np.linspace(0, 1, self.num_tensors))
        
        self.device = device
        self.plot2d_kde = True
        
        return None

    @staticmethod
    def mute_console_output(method):
        def wrapper(*args, **kwargs):
            result = None
            with contextlib.redirect_stdout(io.StringIO()):
                result = method(*args, **kwargs)
            return result
        return wrapper
    
    @mute_console_output
    def _pca(self, dim=2):
        if dim not in [2, 3]:
            raise ValueError("Only 2D and 3D visualizations are supported.")
        
        reshaped_data = [tensor.reshape(tensor.size(0), -1) for tensor in self.tensors]
        combined_data = np.vstack(reshaped_data)
        
        pca = PCA(n_components=dim)
        print("Fitting PCA...", combined_data.shape)
        self.reduced_data = pca.fit_transform(combined_data)
        
        return self.reduced_data
    
    @mute_console_output
    def _tsne(self, dim=2):
        '''
            GPUがある場合、以下の警告が出る？
            OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option.
        '''
        
        if dim not in [2, 3]:
            raise ValueError("Only 2D and 3D visualizations are supported.")
        
        reshaped_data = [tensor.reshape(tensor.size(0), -1) for tensor in self.tensors]
        combined_data = np.vstack(reshaped_data)
        
        tsne = TSNE(n_components=dim, perplexity=10)
        print("Fitting t-SNE...", combined_data.shape)
        self.reduced_data = tsne.fit_transform(combined_data)
        
        return self.reduced_data
    
    @mute_console_output
    def _umap(self, dim=2, random_state=0):
        
        if dim not in [2, 3]:
            raise ValueError("Only 2D and 3D visualizations are supported.")
        
        reshaped_data = [tensor.reshape(tensor.size(0), -1) for tensor in self.tensors]
        combined_data = np.vstack(reshaped_data)
        
        umap = UMAP(n_components=dim, random_state=random_state)
        print("Fitting UMAP...", combined_data.shape)
        self.reduced_data = umap.fit_transform(combined_data)
        
        return self.reduced_data
        

    def pca_plot(self, dim=2, plot_title_append="", fig=None, ax=None):
        
        self._pca(dim)
        plot_title = f"{dim}D PCA of 4D Tensor Data"+plot_title_append
        
        if dim == 2: fig, ax = self._plot_2d(plot_title, fig=fig, ax=ax)
        else:        fig, ax = self._plot_3d(plot_title, fig=fig, ax=ax)
        
        plt.show()
        plt.cla(); plt.clf()
            
        return None

    def tsne_plot(self, dim=2, plot_title_append="", fig=None, ax=None):
        
        self._tsne(dim)
        plot_title = f"{dim}D T-SNE of 4D Tensor Data"+plot_title_append
        
        if dim == 2: fig, ax = self._plot_2d(plot_title, fig=fig, ax=ax)
        else:        fig, ax = self._plot_3d(plot_title, fig=fig, ax=ax)
        
        plt.show()
        plt.cla(); plt.clf()
            
        return None

    def umap_plot(self, dim=2, random_state=0, plot_title_append="", fig=None, ax=None):
        
        self._umap(dim, random_state=random_state)
        plot_title = f"{dim}D T-SNE of 4D Tensor Data"+plot_title_append
        
        if dim == 2: fig, ax = self._plot_2d(plot_title, fig=fig, ax=ax)
        else:        fig, ax = self._plot_3d(plot_title, fig=fig, ax=ax)
        
        plt.show()
        plt.cla(); plt.clf()
            
        return None

    def multi_plot(self, dim=2, methods:list=None):
        
        if methods is None:
            methods = ["pca", "tsne", "umap"]
            
        # fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 6))
        fig = plt.figure(figsize=(6*len(methods), 6))
        axes = [fig.add_subplot(1, len(methods), i+1, projection='3d' if dim==3 else None) for i in range(len(methods))]
        
        if dim == 2: plot_method = lambda plot_title, fig, ax : self._plot_2d(plot_title, fig=fig, ax=ax)
        else:        plot_method = lambda plot_title, fig, ax : self._plot_3d(plot_title, fig=fig, ax=ax)
        
        for i, method in enumerate(methods):
            if method == "pca":
                self._pca(dim)
            if method == "tsne":
                self._tsne(dim)
            if method == "umap":
                self._umap(dim)
                
            _, _ = plot_method(f"{dim}D {method} of 4D Tensor Data", fig=fig, ax=axes[i])
            
        plt.show()
        plt.cla(); plt.clf()
        
        return None

    def plot_2d_3d(self, method:str):
        
        if method is None:
            raise ValueError("methods must be 'pca', 'tsne', or 'umap'.")
            
        # fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 6))
        fig = plt.figure(figsize=(6*2, 6))
        axes = []
        axes.append(fig.add_subplot(1, 2, 1))
        axes.append(fig.add_subplot(1, 2, 2, projection='3d'))
            
        if method == "pca":
            dim_reduction = lambda dim : self._pca(dim)
        if method == "tsne":
            dim_reduction = lambda dim : self._tsne(dim)
        if method == "umap":
            dim_reduction = lambda dim : self._umap(dim)
            
        _ = dim_reduction(2)
        self._plot_2d(f"2D {method}", fig=fig, ax=axes[0])
        _ = dim_reduction(3)
        self._plot_3d(f"3D {method}", fig=fig, ax=axes[1])
            
        plt.show()
        plt.cla(); plt.clf()
        
        return None

    def get_plot_2d_3d(self, method:str, plot_title_append=""):
        
        if method is None:
            raise ValueError("methods must be 'pca', 'tsne', or 'umap'.")
            
        # fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 6))
        fig = plt.figure(figsize=(6*2, 6))
        axes = []
        axes.append(fig.add_subplot(1, 2, 1))
        axes.append(fig.add_subplot(1, 2, 2, projection='3d'))
            
        if method == "pca":
            dim_reduction = lambda dim : self._pca(dim)
        if method == "tsne":
            dim_reduction = lambda dim : self._tsne(dim)
        if method == "umap":
            dim_reduction = lambda dim : self._umap(dim)
            
        _ = dim_reduction(2)
        self._plot_2d(f"2D {method}{plot_title_append}", fig=fig, ax=axes[0])
        _ = dim_reduction(3)
        self._plot_3d(f"3D {method}{plot_title_append}", fig=fig, ax=axes[1])
            
        plot_img = self.figure2ndarray(fig)
        
        return plot_img[:, :, ::-1]
    
    def _plot_2d(self, title="", point_size=10, fig=None, ax=None):
        
        if fig is None and ax is None:
            fig = plt.figure(figsize=(6.0, 5.4))
            ax = fig.add_subplot(111)
        
        colors = self.cmap
        
        split_idx = [sum([len(t) for t in self.tensors[:i+1]]) for i in range(len(self.tensors))][:-1]
        for idx, data in enumerate(np.split(self.reduced_data, split_idx)):
            if self.plot2d_kde:
                sns.kdeplot(x=data[:, 0], y=data[:, 1], fill=True, color=colors[idx%len(colors)]*0.7, alpha=0.1, ax=ax)
            ax.scatter(data[:, 0], data[:, 1], color=colors[idx%len(colors)], label=self.labels[idx], s=point_size)

        ax.set_title(title)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        return fig, ax

    def _plot_3d(self, title="", point_size=10, fig=None, ax=None):
        
        if ax is None:
            fig = plt.figure(figsize=(6.4, 5.8))
            ax = fig.add_subplot(111, projection='3d')
        
        colors = self.cmap
        
        split_idx = [sum([len(t) for t in self.tensors[:i+1]]) for i in range(len(self.tensors))][:-1]
        for idx, data in enumerate(np.split(self.reduced_data, split_idx)):
            ax.scatter(data[:,0], data[:,1], data[:,2], color=colors[idx%len(colors)], label=self.labels[idx], s=point_size)

        ax.set_title(title)
        plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

        return fig, ax
    
    @classmethod
    def figure2ndarray(cls, fig: matplotlib.figure.Figure):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)
        
        # RGBAからRGBに変換
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        return img_array
    
# Inceptionネットワークを使用して特徴ベクトルを抽出
from torchvision.models import inception_v3
def get_Inceptionv3_features(dataset: ImageDataset):
    model = inception_v3(pretrained=True)
    model = model.eval()  # 評価モードに設定
    
    
    dataset.transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3の入力サイズにリサイズ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNetの正規化パラメータ
    ])

    features = []
    for i in range(len(dataset)):
        with torch.no_grad():  # 勾配計算を無効化
            feature = model(dataset[i].unsqueeze(0))
        features.append(feature.squeeze().numpy())
        
    return torch.tensor(features)
    
    
# CLIPネットワークを使用して特徴ベクトルを抽出
import open_clip
def get_CLIP_features(dataset: ImageDataset):
    model, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    dataset.transform = transform

    features = []
    for i in range(len(dataset)):
        with torch.no_grad():  # 勾配計算を無効化
            feature = model.encode_image(dataset[i].unsqueeze(0))
        features.append(feature.squeeze().numpy())
        
    return torch.tensor(features)