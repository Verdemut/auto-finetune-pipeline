import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from torchvision.utils import make_grid


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_path']) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def visualize_dataset(self, dataset, num_samples=8):
        """Визуализация датасета"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(dataset['train']))):
            sample = dataset['train'][i]
            img = sample['pixel_values']
            
            # Денормализация
            img = img / 2 + 0.5
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            caption = sample['caption'][:30] + "..." if len(sample['caption']) > 30 else sample['caption']
            axes[i].set_title(caption)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "dataset_samples.png", dpi=150)
        plt.close()
        print(f"   📊 Визуализация датасета сохранена в {self.output_dir / 'dataset_samples.png'}")
    
    def plot_training_history(self, loss_history):
        """График обучения"""
        if not loss_history:
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        steps = range(len(loss_history))
        ax.plot(steps, loss_history, 'b-', alpha=0.7, linewidth=1)
        
        # Сглаживание
        if len(loss_history) > 10:
            window = min(10, len(loss_history) // 5)
            smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(loss_history)), smoothed, 'r-', linewidth=2, label='Сглаженный')
        
        ax.set_xlabel('Шаг')
        ax.set_ylabel('Loss')
        ax.set_title('Кривая обучения')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curve.png", dpi=150)
        plt.close()
        print(f"   📊 График обучения сохранен в {self.output_dir / 'training_curve.png'}")
    
    def plot_metrics(self, metrics):
        """Визуализация метрик"""
        if not metrics:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Фильтруем только числовые метрики
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        
        if numeric_metrics:
            names = list(numeric_metrics.keys())
            values = list(numeric_metrics.values())
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            bars = ax.bar(names, values, color=colors)
            
            # Добавляем значения на столбцы
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_ylabel('Значение')
            ax.set_title('Метрики качества')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "metrics.png", dpi=150)
            plt.close()
            print(f"   📊 Визуализация метрик сохранена в {self.output_dir / 'metrics.png'}")
    
    def visualize_generations(self, pipe, prompts, output_path):
        """Генерация и визуализация примеров"""
        fig, axes = plt.subplots(1, len(prompts), figsize=(4*len(prompts), 4))
        if len(prompts) == 1:
            axes = [axes]
        
        for i, prompt in enumerate(prompts):
            with torch.no_grad():
                image = pipe(prompt, num_inference_steps=30).images[0]
            
            axes[i].imshow(image)
            axes[i].set_title(prompt[:30] + "..." if len(prompt) > 30 else prompt)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()