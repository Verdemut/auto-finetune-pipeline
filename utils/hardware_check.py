# utils/hardware_check.py - Без эмодзи
import torch
import platform
import psutil
import sys


class HardwareChecker:
    def __init__(self, config):
        self.config = config
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = hasattr(torch, 'backends') and torch.backends.mps.is_available()
        self.device = self._get_device()
        
    def _get_device(self):
        if self.has_cuda:
            return "cuda"
        elif self.has_mps:
            return "mps"
        else:
            return "cpu"
    
    def check(self):
        """Проверка оборудования"""
        print("\n" + "=" * 60)
        print("HARDWARE CHECK")
        print("=" * 60)
        
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Device: {self.device.upper()}")
        
        if self.device == "cuda":
            self._check_cuda()
        elif self.device == "mps":
            self._check_mps()
        else:
            self._check_cpu()
        
        self._print_recommendations()
        
        return self.device
    
    def _check_cuda(self):
        """Проверка CUDA GPU"""
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Определение уровня GPU
        if vram >= 24:
            self.gpu_tier = "ultra"
            print(f"GPU Tier: High-end (24+ GB)")
        elif vram >= 12:
            self.gpu_tier = "high"
            print(f"GPU Tier: Mid-high (12-24 GB)")
        elif vram >= 8:
            self.gpu_tier = "mid"
            print(f"GPU Tier: Mid-range (8-12 GB)")
        elif vram >= 4:
            self.gpu_tier = "low"
            print(f"GPU Tier: Entry-level (4-8 GB)")
        else:
            self.gpu_tier = "minimal"
            print(f"GPU Tier: Minimal (<4 GB)")
    
    def _check_mps(self):
        """Проверка Apple Silicon"""
        print(f"Device: Apple Silicon (MPS)")
        print(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        self.gpu_tier = "apple"
    
    def _check_cpu(self):
        """Проверка CPU"""
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"System RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        self.gpu_tier = "cpu"
    
    def _print_recommendations(self):
        """Рекомендации на основе оборудования"""
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        
        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram >= 24:
                print("Status: Excellent - Full fine-tuning is possible")
                print("Recommended batch_size: 4-8")
            elif vram >= 12:
                print("Status: Good - Suitable for LoRA and DreamBooth")
                print("Recommended batch_size: 2-4")
            elif vram >= 8:
                print("Status: Acceptable - LoRA recommended")
                print("Recommended batch_size: 1-2")
                print("Enable: attention_slicing and fp16")
            else:
                print("Status: Limited - Use LoRA with batch_size=1")
                print("Enable: attention_slicing, fp16, and xformers")
        
        elif self.device == "mps":
            print("Status: Apple Silicon detected")
            print("Recommended method: LoRA")
            print("Recommended batch_size: 1")
            print("Note: Use float32 for stability")
        
        else:
            print("Status: GPU not detected - Training will be very slow")
            print("Recommendation: Use Google Colab or cloud services")
            print("Or reduce num_epochs to 2-3 for testing")
    
    def get_optimal_batch_size(self):
        """Автоматический выбор batch_size"""
        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram >= 24:
                return 4
            elif vram >= 16:
                return 2
            elif vram >= 12:
                return 2
            elif vram >= 8:
                return 1
            else:
                return 1
        else:
            return 1
    
    def get_optimal_gradient_accumulation(self, batch_size):
        """Автоматический выбор gradient accumulation"""
        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram >= 24:
                return 1
            elif vram >= 12:
                return 2 if batch_size == 1 else 1
            else:
                return 4 if batch_size == 1 else 2
        else:
            return 4
    
    def get_optimal_method(self, dataset_size):
        """Выбор метода обучения"""
        if dataset_size < 50:
            return "textual_inversion"
        elif dataset_size < 500:
            return "lora"
        else:
            return "dreambooth"
    
    def get_optimal_learning_rate(self, method):
        """Автоматический выбор learning rate"""
        rates = {
            "lora": 1e-4,
            "dreambooth": 5e-6,
            "textual_inversion": 5e-4
        }
        return rates.get(method, 1e-4)
    
    def get_optimal_fp16(self):
        """Определить можно ли использовать fp16"""
        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return vram < 16
        return False
    
    def get_optimal_num_epochs(self, dataset_size, method):
        """Автоматический выбор количества эпох"""
        base_epochs = {
            "lora": 100,
            "dreambooth": 50,
            "textual_inversion": 200
        }
        
        epochs = base_epochs.get(method, 100)
        
        if dataset_size < 50:
            epochs = int(epochs * 1.5)
        elif dataset_size > 500:
            epochs = int(epochs * 0.7)
        
        return min(epochs, 200)
    
    def estimate_training_time(self, dataset_size, num_epochs, batch_size):
        """Оценка времени обучения"""
        steps_per_epoch = dataset_size / batch_size
        total_steps = steps_per_epoch * num_epochs
        
        # Примерная скорость на разных устройствах (сек на батч)
        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram >= 16:
                speed = 0.3
            elif vram >= 8:
                speed = 0.5
            else:
                speed = 1.0
        elif self.device == "mps":
            speed = 1.5
        else:
            speed = 30.0
        
        estimated_seconds = total_steps * speed
        estimated_minutes = estimated_seconds / 60
        estimated_hours = estimated_minutes / 60
        
        return {
            "seconds": estimated_seconds,
            "minutes": estimated_minutes,
            "hours": estimated_hours,
            "speed_per_batch": speed
        }
    
    def print_training_estimate(self, dataset_size, num_epochs, batch_size):
        """Вывод оценки времени"""
        estimate = self.estimate_training_time(dataset_size, num_epochs, batch_size)
        
        print(f"\nTRAINING TIME ESTIMATE:")
        print(f"  Batches per epoch: {dataset_size / batch_size:.0f}")
        print(f"  Total steps: {dataset_size / batch_size * num_epochs:.0f}")
        print(f"  Estimated time: ", end="")
        
        if estimate["hours"] >= 1:
            print(f"{estimate['hours']:.1f} hours")
        elif estimate["minutes"] >= 1:
            print(f"{estimate['minutes']:.0f} minutes")
        else:
            print(f"{estimate['seconds']:.0f} seconds")
        
        return estimate