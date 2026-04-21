# webui_platform.py - Многоязычный интерфейс с выбором платформы

import gradio as gr
import torch
from pathlib import Path
import shutil
import os
import json
from datetime import datetime
import threading
import pandas as pd
from PIL import Image
import yaml
import psutil
import subprocess
import sys
import zipfile
import tempfile

# Импортируем наш пайплайн
from main import AutoFinetunePipeline
from utils.hardware_check import HardwareChecker


class PlatformAwareGUI:
    def __init__(self, language="en"):
        self.language = language
        self.dataset_path = None
        self.output_path = None
        self.training_thread = None
        self.is_training = False
        self.current_config = None
        self.log_messages = []
        self.current_platform = "local"  # local or colab
        
        # Language dictionaries (сокращенные для краткости, но можно расширить)
        self.strings = {
            "en": {
                "title": "Auto-Finetune Pipeline",
                "subtitle": "Professional Neural Network Training Interface",
                "platform_select": "Training Platform",
                "platform_local": "Local PC (GPU/CPU)",
                "platform_colab": "Google Colab (Cloud GPU)",
                "platform_info": "Select where to run the training",
                "colab_info": "Google Colab offers free GPU (Tesla T4, V100, A100). Upload your dataset and run the notebook.",
                "local_info": "Train on your local machine using your GPU or CPU.",
                
                "tab_dataset": "Dataset Management",
                "tab_config": "Training Configuration",
                "tab_advanced": "Advanced Settings",
                "tab_control": "Training Control",
                "tab_monitor": "Monitoring",
                "tab_inference": "Inference",
                "tab_colab": "Google Colab Export",
                
                # Colab tab
                "colab_title": "Export to Google Colab",
                "colab_description": "Generate a Colab notebook to train your model in the cloud",
                "dataset_export": "Export Dataset",
                "export_dataset_btn": "Export Dataset as ZIP",
                "export_config_btn": "Export Config as JSON",
                "generate_notebook_btn": "Generate Colab Notebook",
                "download_notebook": "Download Notebook (.ipynb)",
                "colab_instructions": "Instructions",
                "colab_step1": "1. Upload the dataset ZIP to Google Drive or Colab",
                "colab_step2": "2. Upload the config JSON",
                "colab_step3": "3. Run the notebook cells in order",
                "colab_step4": "4. Download your trained model from Colab",
                "colab_note": "Note: Colab provides free GPU for ~12 hours. Save your model before disconnection.",
                
                # Rest of strings (keep existing)
                "dataset_upload": "Dataset Upload",
                "upload_images": "Upload Images",
                "default_caption": "Default Caption",
                "apply_caption": "Apply Caption to All",
                "save_dataset": "Save Dataset",
                "dataset_name": "Dataset Name",
                "dataset_status": "No dataset selected",
                "image_preview": "Image Preview",
                "caption_editor": "Caption Editor",
                "filename": "Filename",
                "caption": "Caption",
                
                "basic_params": "Basic Training Parameters",
                "training_method": "Training Method",
                "num_epochs": "Number of Epochs",
                "batch_size": "Batch Size",
                "grad_accum": "Gradient Accumulation Steps",
                "learning_rate": "Learning Rate",
                "lr_scheduler": "Learning Rate Scheduler",
                "warmup_steps": "Warmup Steps",
                "weight_decay": "Weight Decay",
                "data_processing": "Data Processing",
                "image_resolution": "Image Resolution",
                "validation_split": "Validation Split",
                "output_name": "Model Output Name",
                
                "model_arch": "Model Architecture Settings",
                "base_model": "Base Model",
                "fallback_models": "Fallback Models",
                "lora_params": "LoRA Parameters",
                "lora_rank": "LoRA Rank",
                "lora_alpha": "LoRA Alpha",
                "lora_modules": "Target Modules",
                "memory_opt": "Memory Optimization",
                "use_fp16": "Use FP16",
                "use_xformers": "Enable xFormers",
                "attention_slicing": "Attention Slicing",
                "vae_slicing": "VAE Slicing",
                "advanced_train": "Advanced Training Settings",
                "grad_checkpoint": "Gradient Checkpointing",
                "max_grad_norm": "Max Gradient Norm",
                "random_seed": "Random Seed",
                
                "control_panel": "Training Control Panel",
                "start_training": "START TRAINING",
                "stop_training": "STOP TRAINING",
                "status_waiting": "Status: Waiting for start",
                "resume_checkpoint": "Resume from Checkpoint",
                "training_logs": "Training Logs",
                
                "monitor_title": "Training Monitoring",
                "refresh_metrics": "Refresh Metrics",
                "waiting_metrics": "Waiting for training to start...",
                "metrics_table": "Training Metrics",
                "metric_last_loss": "Last Loss",
                "metric_avg_loss": "Average Loss (10 steps)",
                "metric_min_loss": "Minimum Loss",
                "metric_total_steps": "Total Steps",
                
                "inference_title": "Image Generation",
                "select_model": "Select Model",
                "prompt": "Prompt",
                "negative_prompt": "Negative Prompt",
                "inference_steps": "Inference Steps",
                "guidance_scale": "Guidance Scale",
                "generate_btn": "Generate Image",
                "refresh_models": "Refresh Model List",
                "generated_image": "Generated Image",
                
                # Status messages
                "status_saved": "Dataset saved: {path}\nTotal images: {count}",
                "status_no_images": "Error: No images to save",
                "status_training_started": "Status: Training started",
                "status_training_stopped": "Status: Training stopped by user",
                "status_training_running": "Status: Training already running",
                "status_dataset_not_found": "Error: Dataset not found",
                "status_model_selected": "Error: No model selected",
                "status_prompt_empty": "Error: Please enter a prompt",
                "status_generation_success": "Generation completed successfully",
                "status_generation_error": "Error: {error}",
                "status_export_success": "Export successful: {path}",
                "status_export_error": "Export error: {error}",
                
                "hardware_status": "System Status",
                "hardware_cpu": "CPU",
                "hardware_ram": "RAM",
                "hardware_gpu": "GPU",
                "hardware_cuda": "CUDA",
                "hardware_pytorch": "PyTorch",
                "hardware_not_detected": "Not detected",
                
                "language": "Language",
                "lang_en": "English",
                "lang_ru": "Russian",
                "restart_info": "Language will change after restart",
                "restart_button": "Apply and Restart",
            },
            "ru": {
                "title": "Auto-Finetune Pipeline",
                "subtitle": "Профессиональный интерфейс обучения нейросетей",
                "platform_select": "Платформа для обучения",
                "platform_local": "Локальный ПК (GPU/CPU)",
                "platform_colab": "Google Colab (Облачный GPU)",
                "platform_info": "Выберите, где будет выполняться обучение",
                "colab_info": "Google Colab предоставляет бесплатный GPU (Tesla T4, V100, A100). Загрузите датасет и запустите ноутбук.",
                "local_info": "Обучение на вашем локальном компьютере с использованием GPU или CPU.",
                
                "tab_dataset": "Управление датасетом",
                "tab_config": "Настройки обучения",
                "tab_advanced": "Расширенные настройки",
                "tab_control": "Управление обучением",
                "tab_monitor": "Мониторинг",
                "tab_inference": "Генерация",
                "tab_colab": "Экспорт в Colab",
                
                # Colab tab
                "colab_title": "Экспорт в Google Colab",
                "colab_description": "Сгенерируйте Colab ноутбук для обучения модели в облаке",
                "dataset_export": "Экспорт датасета",
                "export_dataset_btn": "Экспортировать датасет в ZIP",
                "export_config_btn": "Экспортировать конфиг в JSON",
                "generate_notebook_btn": "Сгенерировать Colab ноутбук",
                "download_notebook": "Скачать ноутбук (.ipynb)",
                "colab_instructions": "Инструкция",
                "colab_step1": "1. Загрузите ZIP с датасетом в Google Drive или Colab",
                "colab_step2": "2. Загрузите JSON с конфигурацией",
                "colab_step3": "3. Запустите ячейки ноутбука по порядку",
                "colab_step4": "4. Скачайте обученную модель из Colab",
                "colab_note": "Примечание: Colab предоставляет бесплатный GPU на ~12 часов. Сохраняйте модель перед отключением.",
                
                "dataset_upload": "Загрузка датасета",
                "upload_images": "Загрузить изображения",
                "default_caption": "Описание по умолчанию",
                "apply_caption": "Применить описание ко всем",
                "save_dataset": "Сохранить датасет",
                "dataset_name": "Название датасета",
                "dataset_status": "Датасет не выбран",
                "image_preview": "Предпросмотр изображений",
                "caption_editor": "Редактор описаний",
                "filename": "Имя файла",
                "caption": "Описание",
                
                "basic_params": "Основные параметры обучения",
                "training_method": "Метод обучения",
                "num_epochs": "Количество эпох",
                "batch_size": "Размер батча",
                "grad_accum": "Шаги накопления градиента",
                "learning_rate": "Скорость обучения",
                "lr_scheduler": "Планировщик скорости",
                "warmup_steps": "Шаги разогрева",
                "weight_decay": "Регуляризация весов",
                "data_processing": "Обработка данных",
                "image_resolution": "Разрешение изображений",
                "validation_split": "Доля валидации",
                "output_name": "Название модели",
                
                "model_arch": "Архитектура модели",
                "base_model": "Базовая модель",
                "fallback_models": "Резервные модели",
                "lora_params": "Параметры LoRA",
                "lora_rank": "Ранг LoRA",
                "lora_alpha": "Альфа LoRA",
                "lora_modules": "Целевые модули",
                "memory_opt": "Оптимизация памяти",
                "use_fp16": "Использовать FP16",
                "use_xformers": "Включить xFormers",
                "attention_slicing": "Разделение внимания",
                "vae_slicing": "Разделение VAE",
                "advanced_train": "Расширенные настройки",
                "grad_checkpoint": "Контрольные точки градиентов",
                "max_grad_norm": "Макс. норма градиента",
                "random_seed": "Случайное зерно",
                
                "control_panel": "Панель управления",
                "start_training": "НАЧАТЬ ОБУЧЕНИЕ",
                "stop_training": "ОСТАНОВИТЬ",
                "status_waiting": "Статус: Ожидание",
                "resume_checkpoint": "Возобновить из чекпоинта",
                "training_logs": "Логи обучения",
                
                "monitor_title": "Мониторинг",
                "refresh_metrics": "Обновить",
                "waiting_metrics": "Ожидание...",
                "metrics_table": "Метрики",
                "metric_last_loss": "Последний loss",
                "metric_avg_loss": "Средний loss",
                "metric_min_loss": "Минимальный loss",
                "metric_total_steps": "Всего шагов",
                
                "inference_title": "Генерация",
                "select_model": "Выберите модель",
                "prompt": "Промпт",
                "negative_prompt": "Негативный промпт",
                "inference_steps": "Шаги генерации",
                "guidance_scale": "Масштаб",
                "generate_btn": "Сгенерировать",
                "refresh_models": "Обновить список",
                "generated_image": "Результат",
                
                "status_saved": "Датасет сохранен: {path}\nВсего изображений: {count}",
                "status_no_images": "Ошибка: Нет изображений",
                "status_training_started": "Статус: Обучение запущено",
                "status_training_stopped": "Статус: Обучение остановлено",
                "status_training_running": "Статус: Обучение уже запущено",
                "status_dataset_not_found": "Ошибка: Датасет не найден",
                "status_model_selected": "Ошибка: Модель не выбрана",
                "status_prompt_empty": "Ошибка: Введите промпт",
                "status_generation_success": "Генерация завершена",
                "status_generation_error": "Ошибка: {error}",
                "status_export_success": "Экспорт успешен: {path}",
                "status_export_error": "Ошибка экспорта: {error}",
                
                "hardware_status": "Состояние системы",
                "hardware_cpu": "Процессор",
                "hardware_ram": "ОЗУ",
                "hardware_gpu": "Видеокарта",
                "hardware_cuda": "CUDA",
                "hardware_pytorch": "PyTorch",
                "hardware_not_detected": "Не обнаружена",
                
                "language": "Язык",
                "lang_en": "Английский",
                "lang_ru": "Русский",
                "restart_info": "Язык изменится после перезапуска",
                "restart_button": "Применить и перезапустить",
            }
        }
        
        self.hardware = HardwareChecker({})
        self.device = self.hardware.check()
    
    def _(self, key, **kwargs):
        text = self.strings[self.language].get(key, key)
        if kwargs:
            text = text.format(**kwargs)
        return text
    
    def create_interface(self):
        with gr.Blocks(title="Auto-Finetune Pipeline") as demo:
            # Platform selector at the top
            with gr.Row():
                with gr.Column(scale=6):
                    gr.Markdown(f"# {self._('title')}")
                    gr.Markdown(f"### {self._('subtitle')}")
                with gr.Column(scale=2):
                    lang_selector = gr.Dropdown(
                        choices=[self._("lang_en"), self._("lang_ru")],
                        value=self._("lang_en") if self.language == "en" else self._("lang_ru"),
                        label=self._("language")
                    )
                    restart_btn = gr.Button(self._("restart_button"), variant="secondary", size="sm")
            
            # Platform selection
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### {self._('platform_select')}")
                    platform_selector = gr.Radio(
                        choices=[self._("platform_local"), self._("platform_colab")],
                        value=self._("platform_local"),
                        label=self._("platform_select"),
                        info=self._("platform_info")
                    )
                    platform_info = gr.Markdown(self._("local_info"))
            
            # Status bar
            with gr.Row():
                with gr.Column():
                    status_display = gr.Markdown(self._get_hardware_status())
            
            # Main content
            with gr.Tabs():
                with gr.TabItem(self._("tab_dataset")):
                    self._create_dataset_ui()
                
                with gr.TabItem(self._("tab_config")):
                    self._create_training_ui()
                
                with gr.TabItem(self._("tab_advanced")):
                    self._create_advanced_ui()
                
                # Conditional content based on platform
                with gr.TabItem(self._("tab_control")):
                    self._create_control_ui()
                
                with gr.TabItem(self._("tab_monitor")):
                    self._create_monitor_ui()
                
                with gr.TabItem(self._("tab_inference")):
                    self._create_inference_ui()
                
                with gr.TabItem(self._("tab_colab")):
                    self._create_colab_export_ui()
            
            # Platform change handler
            def on_platform_change(platform):
                if platform == self._("platform_local"):
                    return self._("local_info"), gr.update(visible=True), gr.update(visible=False)
                else:
                    return self._("colab_info"), gr.update(visible=False), gr.update(visible=True)
            
            platform_selector.change(
                on_platform_change,
                inputs=[platform_selector],
                outputs=[platform_info, status_display, status_display]
            )
            
            # Restart function
            def restart_app(lang_selected):
                new_lang = "ru" if lang_selected == self._("lang_ru") else "en"
                with open("language_pref.txt", "w") as f:
                    f.write(new_lang)
                python = sys.executable
                subprocess.Popen([python, __file__])
                os._exit(0)
            
            restart_btn.click(
                restart_app,
                inputs=[lang_selector],
                outputs=[]
            )
        
        return demo
    
    def _get_hardware_status(self):
        status = f"## {self._('hardware_status')}\n\n"
        
        status += f"**{self._('hardware_cpu')}**: {psutil.cpu_count()} cores\n"
        status += f"**{self._('hardware_ram')}**: {psutil.virtual_memory().total / 1e9:.1f} GB\n"
        
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            status += f"**{self._('hardware_gpu')}**: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)\n"
            status += f"**{self._('hardware_cuda')}**: {torch.version.cuda}\n"
            status += f"**{self._('hardware_pytorch')}**: {torch.__version__}\n"
        else:
            status += f"**{self._('hardware_gpu')}**: {self._('hardware_not_detected')}\n"
        
        return status
    
    def _create_colab_export_ui(self):
        """Create Google Colab export interface"""
        
        gr.Markdown(f"### {self._('colab_title')}")
        gr.Markdown(self._("colab_description"))
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"#### {self._('dataset_export')}")
                
                self.export_dataset_btn = gr.Button(self._("export_dataset_btn"), variant="secondary")
                self.export_status = gr.Markdown("")
                
                self.export_config_btn = gr.Button(self._("export_config_btn"), variant="secondary")
                self.config_status = gr.Markdown("")
            
            with gr.Column(scale=2):
                gr.Markdown(f"#### {self._('generate_notebook_btn')}")
                
                self.generate_btn = gr.Button(self._("generate_notebook_btn"), variant="primary")
                self.notebook_output = gr.File(label=self._("download_notebook"))
                
                gr.Markdown(f"#### {self._('colab_instructions')}")
                gr.Markdown(f"""
                {self._('colab_step1')}
                {self._('colab_step2')}
                {self._('colab_step3')}
                {self._('colab_step4')}
                
                {self._('colab_note')}
                """)
        
        def export_dataset(dataset_name):
            dataset_path = Path(f"./datasets/{dataset_name}")
            if not dataset_path.exists():
                return self._("status_dataset_not_found")
            
            try:
                zip_path = Path(f"./exports/{dataset_name}.zip")
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in dataset_path.rglob("*"):
                        zipf.write(file, file.relative_to(dataset_path.parent))
                
                return self._("status_export_success", path=str(zip_path))
            except Exception as e:
                return self._("status_export_error", error=str(e))
        
        def export_config():
            # Collect current config
            config = {
                "dataset_name": self.dataset_name.value if hasattr(self, 'dataset_name') else "my_dataset",
                "output_name": self.output_name.value if hasattr(self, 'output_name') else "my_model",
                "method": self.method.value if hasattr(self, 'method') else "auto",
                "num_epochs": self.num_epochs.value if hasattr(self, 'num_epochs') else 50,
                "batch_size": self.batch_size.value if hasattr(self, 'batch_size') else 1,
                "gradient_accumulation": self.gradient_accumulation.value if hasattr(self, 'gradient_accumulation') else 2,
                "learning_rate": self.learning_rate.value if hasattr(self, 'learning_rate') else "1e-4",
                "image_size": self.image_size.value if hasattr(self, 'image_size') else 512,
                "base_model": self.base_model.value if hasattr(self, 'base_model') else "runwayml/stable-diffusion-v1-5",
                "use_fp16": self.use_fp16.value if hasattr(self, 'use_fp16') else True,
                "enable_xformers": self.enable_xformers.value if hasattr(self, 'enable_xformers') else True,
            }
            
            try:
                config_path = Path("./exports/training_config.json")
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                return self._("status_export_success", path=str(config_path))
            except Exception as e:
                return self._("status_export_error", error=str(e))
        
        def generate_colab_notebook():
            notebook_content = self._create_colab_notebook()
            
            notebook_path = Path("./exports/auto_finetune_colab.ipynb")
            notebook_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(notebook_content)
            
            return notebook_path
        
        self.export_dataset_btn.click(
            export_dataset,
            inputs=[self.dataset_name],
            outputs=[self.export_status]
        )
        
        self.export_config_btn.click(
            export_config,
            outputs=[self.config_status]
        )
        
        self.generate_btn.click(
            generate_colab_notebook,
            outputs=[self.notebook_output]
        )
    
    def _create_colab_notebook(self):
        """Generate Colab notebook content"""
        
        return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {"id": "header"},
   "source": [
    "# Auto-Finetune Pipeline for Google Colab\\n",
    "## Professional Neural Network Training in the Cloud\\n",
    "\\n",
    "This notebook will train a Stable Diffusion model on your custom dataset using Google's free GPU.\\n",
    "\\n",
    "**Estimated time**: 10-60 minutes depending on dataset size and settings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "setup"},
   "source": [
    "## 1. Setup Environment\\n",
    "\\n",
    "First, let's check if GPU is available and install required packages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "check_gpu"},
   "source": [
    "import torch\\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\\n",
    "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\\n",
    "else:\\n",
    "    print(\"WARNING: GPU not found! Enable GPU in Runtime > Change runtime type > GPU\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "install"},
   "source": [
    "!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\\n",
    "!pip install -q diffusers transformers accelerate datasets\\n",
    "!pip install -q gradio pillow numpy pandas\\n",
    "!pip install -q scipy tensorboard\\n",
    "!pip install -q huggingface_hub pyyaml tqdm psutil\\n",
    "!pip install -q xformers\\n",
    "\\n",
    "print(\"All packages installed successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "upload"},
   "source": [
    "## 2. Upload Your Dataset\\n",
    "\\n",
    "Upload the dataset ZIP file and config JSON you exported from the web interface."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "upload_files"},
   "source": [
    "from google.colab import files\\n",
    "import zipfile\\n",
    "import json\\n",
    "import os\\n",
    "from pathlib import Path\\n",
    "\\n",
    "print(\"Upload dataset ZIP file...\")\\n",
    "uploaded = files.upload()\\n",
    "\\n",
    "for filename in uploaded.keys():\\n",
    "    if filename.endswith('.zip'):\\n",
    "        with zipfile.ZipFile(filename, 'r') as zip_ref:\\n",
    "            zip_ref.extractall('./dataset')\\n",
    "        print(f\"Dataset extracted to ./dataset\")\\n",
    "\\n",
    "print(\"\\nUpload config JSON file...\")\\n",
    "uploaded = files.upload()\\n",
    "\\n",
    "config = None\\n",
    "for filename in uploaded.keys():\\n",
    "    if filename.endswith('.json'):\\n",
    "        with open(filename, 'r') as f:\\n",
    "            config = json.load(f)\\n",
    "        print(f\"Config loaded from {filename}\")\\n",
    "\\n",
    "print(\"\\nDataset and config loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "config"},
   "source": [
    "## 3. Training Configuration\\n",
    "\\n",
    "Here are your training parameters:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "show_config"},
   "source": [
    "if config:\\n",
    "    print(\"Training Configuration:\")\\n",
    "    for key, value in config.items():\\n",
    "        print(f\"  {key}: {value}\")\\n",
    "else:\\n",
    "    print(\"Using default configuration\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "train"},
   "source": [
    "## 4. Train the Model\\n",
    "\\n",
    "This cell will start the actual training process. It may take 10-60 minutes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "training"},
   "source": [
    "import sys\\n",
    "import torch\\n",
    "from diffusers import StableDiffusionPipeline, DDPMScheduler\\n",
    "from torch.utils.data import Dataset, DataLoader\\n",
    "from torchvision import transforms\\n",
    "from PIL import Image\\n",
    "import pandas as pd\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Configuration\\n",
    "DATASET_PATH = Path(\"./dataset\")\\n",
    "OUTPUT_PATH = Path(\"./outputs/model\")\\n",
    "OUTPUT_PATH.mkdir(parents=True, exist_ok=True)\\n",
    "\\n",
    "# Find dataset folder\\n",
    "dataset_folder = None\\n",
    "for item in DATASET_PATH.iterdir():\\n",
    "    if item.is_dir():\\n",
    "        dataset_folder = item\\n",
    "        break\\n",
    "\\n",
    "if dataset_folder is None:\\n",
    "    dataset_folder = DATASET_PATH\\n",
    "\\n",
    "print(f\"Dataset folder: {dataset_folder}\")\\n",
    "\\n",
    "# Dataset class\\n",
    "class SimpleDataset(Dataset):\\n",
    "    def __init__(self, folder):\\n",
    "        self.images = list(folder.glob(\"*.jpg\")) + list(folder.glob(\"*.png\"))\\n",
    "        self.transform = transforms.Compose([\\n",
    "            transforms.Resize((512, 512)),\\n",
    "            transforms.ToTensor(),\\n",
    "            transforms.Normalize([0.5], [0.5])\\n",
    "        ])\\n",
    "        csv_path = folder / \"captions.csv\"\\n",
    "        if csv_path.exists():\\n",
    "            df = pd.read_csv(csv_path)\\n",
    "            self.captions = dict(zip(df['image'].astype(str), df['caption']))\\n",
    "        else:\\n",
    "            self.captions = {}\\n",
    "    \\n",
    "    def __len__(self):\\n",
    "        return len(self.images)\\n",
    "    \\n",
    "    def __getitem__(self, idx):\\n",
    "        img = Image.open(self.images[idx]).convert('RGB')\\n",
    "        img = self.transform(img)\\n",
    "        name = self.images[idx].stem\\n",
    "        caption = self.captions.get(name, f\"image_{name}\")\\n",
    "        return {\"pixel_values\": img, \"caption\": caption}\\n",
    "\\n",
    "# Load dataset\\n",
    "dataset = SimpleDataset(dataset_folder)\\n",
    "dataloader = DataLoader(dataset, batch_size=1, shuffle=True)\\n",
    "print(f\"Loaded {len(dataset)} images\")\\n",
    "\\n",
    "# Load model\\n",
    "print(\"Loading Stable Diffusion model...\")\\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\\n",
    "pipe = StableDiffusionPipeline.from_pretrained(\\n",
    "    \"runwayml/stable-diffusion-v1-5\",\\n",
    "    torch_dtype=torch.float32,\\n",
    "    safety_checker=None\\n",
    ").to(device)\\n",
    "\\n",
    "# Freeze components\\n",
    "for param in pipe.text_encoder.parameters():\\n",
    "    param.requires_grad = False\\n",
    "for param in pipe.vae.parameters():\\n",
    "    param.requires_grad = False\\n",
    "\\n",
    "# Optimizer\\n",
    "optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)\\n",
    "noise_scheduler = DDPMScheduler.from_pretrained(\\n",
    "    \"runwayml/stable-diffusion-v1-5\",\\n",
    "    subfolder=\"scheduler\"\\n",
    ")\\n",
    "\\n",
    "# Training\\n",
    "num_epochs = config.get('num_epochs', 50) if config else 50\\n",
    "print(f\"\\nStarting training for {num_epochs} epochs...\")\\n",
    "\\n",
    "for epoch in range(num_epochs):\\n",
    "    total_loss = 0\\n",
    "    for batch in dataloader:\\n",
    "        pixel_values = batch[\"pixel_values\"].to(device)\\n",
    "        captions = batch[\"caption\"]\\n",
    "        \\n",
    "        # Encode text\\n",
    "        text_inputs = pipe.tokenizer(\\n",
    "            captions,\\n",
    "            padding=\"max_length\",\\n",
    "            max_length=pipe.tokenizer.model_max_length,\\n",
    "            truncation=True,\\n",
    "            return_tensors=\"pt\"\\n",
    "        ).to(device)\\n",
    "        \\n",
    "        with torch.no_grad():\\n",
    "            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]\\n",
    "            latents = pipe.vae.encode(pixel_values).latent_dist.sample()\\n",
    "            latents = latents * pipe.vae.config.scaling_factor\\n",
    "        \\n",
    "        # Add noise\\n",
    "        noise = torch.randn_like(latents)\\n",
    "        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,\\n",
    "                                  (latents.shape[0],), device=device).long()\\n",
    "        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)\\n",
    "        \\n",
    "        # Predict noise\\n",
    "        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample\\n",
    "        loss = torch.nn.functional.mse_loss(noise_pred, noise)\\n",
    "        \\n",
    "        # Backward\\n",
    "        optimizer.zero_grad()\\n",
    "        loss.backward()\\n",
    "        optimizer.step()\\n",
    "        \\n",
    "        total_loss += loss.item()\\n",
    "    \\n",
    "    avg_loss = total_loss / len(dataloader)\\n",
    "    print(f\"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}\")\\n",
    "\\n",
    "# Save model\\n",
    "print(\"\\nSaving model...\")\\n",
    "pipe.unet.save_pretrained(OUTPUT_PATH / \"unet\")\\n",
    "print(f\"Model saved to {OUTPUT_PATH}\")\\n",
    "\\n",
    "# Pack model for download\\n",
    "!zip -r /content/model.zip /content/outputs/\\n",
    "print(\"\\nModel packed to model.zip\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "download"},
   "source": [
    "## 5. Download Your Trained Model\\n",
    "\\n",
    "Click the download button below to save your trained model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "download_model"},
   "source": [
    "from google.colab import files\\n",
    "files.download(\"model.zip\")\\n",
    "print(\"Download started!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {"id": "test"},
   "source": [
    "## 6. Test Your Model\\n",
    "\\n",
    "Optional: Test the model with a sample prompt."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"id": "test_generation"},
   "source": [
    "def generate_image(prompt):\\n",
    "    pipe = StableDiffusionPipeline.from_pretrained(\\n",
    "        OUTPUT_PATH,\\n",
    "        torch_dtype=torch.float32\\n",
    "    ).to(device)\\n",
    "    \\n",
    "    with torch.no_grad():\\n",
    "        image = pipe(prompt, num_inference_steps=30).images[0]\\n",
    "    return image\\n",
    "\\n",
    "# Example\\n",
    "image = generate_image(\"test image\")\\n",
    "image.save(\"test_output.png\")\\n",
    "print(\"Test image saved as test_output.png\")"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {"provenance": []},
  "kernelspec": {"display_name": "Python 3", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
'''
    
    def _create_dataset_ui(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### {self._('dataset_upload')}")
                
                self.image_upload = gr.File(
                    label=self._("upload_images"),
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"]
                )
                
                self.caption_text = gr.Textbox(
                    label=self._("default_caption"),
                    placeholder=self._("default_caption_placeholder"),
                    lines=2
                )
                
                with gr.Row():
                    apply_caption_btn = gr.Button(self._("apply_caption"), variant="secondary")
                    save_dataset_btn = gr.Button(self._("save_dataset"), variant="primary")
                
                self.dataset_name = gr.Textbox(
                    label=self._("dataset_name"),
                    value="my_dataset"
                )
                
                self.dataset_status = gr.Markdown(self._("dataset_status"))
            
            with gr.Column(scale=2):
                self.gallery = gr.Gallery(
                    label=self._("image_preview"),
                    columns=4,
                    rows=2,
                    height=400
                )
                
                gr.Markdown(f"### {self._('caption_editor')}")
                self.caption_editor = gr.Dataframe(
                    headers=[self._("filename"), self._("caption")],
                    label=self._("caption_editor"),
                    interactive=True,
                    wrap=True
                )
        
        def load_images_to_gallery(files):
            if not files:
                return [], pd.DataFrame(columns=[self._("filename"), self._("caption")])
            
            images = []
            data = []
            for file in files:
                try:
                    img = Image.open(file.name)
                    images.append(img)
                    name = Path(file.name).stem
                    data.append([name, f"image_{name}"])
                except:
                    continue
            return images, pd.DataFrame(data, columns=[self._("filename"), self._("caption")])
        
        def apply_caption_to_all(files, caption):
            if not files:
                return pd.DataFrame(columns=[self._("filename"), self._("caption")])
            data = [[Path(f.name).stem, caption if caption else f"image_{Path(f.name).stem}"] for f in files]
            return pd.DataFrame(data, columns=[self._("filename"), self._("caption")])
        
        def save_dataset(files, dataframe, dataset_name):
            if not files:
                return self._("status_no_images")
            dataset_path = Path(f"./datasets/{dataset_name}")
            dataset_path.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy(Path(file.name), dataset_path / Path(file.name).name)
            if dataframe is not None and len(dataframe) > 0:
                df = pd.DataFrame(dataframe.values, columns=["image", "caption"])
                df.to_csv(dataset_path / "captions.csv", index=False, encoding='utf-8')
            return self._("status_saved", path=str(dataset_path), count=len(files))
        
        self.image_upload.change(load_images_to_gallery, [self.image_upload], [self.gallery, self.caption_editor])
        apply_caption_btn.click(apply_caption_to_all, [self.image_upload, self.caption_text], [self.caption_editor])
        save_dataset_btn.click(save_dataset, [self.image_upload, self.caption_editor, self.dataset_name], [self.dataset_status])
    
    def _create_training_ui(self):
        gr.Markdown(f"### {self._('basic_params')}")
        with gr.Row():
            with gr.Column():
                self.method = gr.Dropdown(label=self._("training_method"), choices=["auto", "lora", "dreambooth", "textual_inversion"], value="auto")
                self.num_epochs = gr.Slider(label=self._("num_epochs"), minimum=1, maximum=200, value=50, step=1)
                self.batch_size = gr.Slider(label=self._("batch_size"), minimum=1, maximum=8, value=1, step=1)
                self.gradient_accumulation = gr.Slider(label=self._("grad_accum"), minimum=1, maximum=8, value=2, step=1)
            with gr.Column():
                self.learning_rate = gr.Textbox(label=self._("learning_rate"), value="1e-4")
                self.lr_scheduler = gr.Dropdown(label=self._("lr_scheduler"), choices=["cosine", "linear", "constant"], value="cosine")
                self.warmup_steps = gr.Slider(label=self._("warmup_steps"), minimum=0, maximum=1000, value=100, step=10)
                self.weight_decay = gr.Textbox(label=self._("weight_decay"), value="0.01")
        
        gr.Markdown(f"### {self._('data_processing')}")
        with gr.Row():
            self.image_size = gr.Dropdown(label=self._("image_resolution"), choices=[256, 384, 512, 768], value=512)
            self.validation_split = gr.Slider(label=self._("validation_split"), minimum=0, maximum=0.3, value=0.1, step=0.05)
        self.output_name = gr.Textbox(label=self._("output_name"), value="my_model")
    
    def _create_advanced_ui(self):
        gr.Markdown(f"### {self._('model_arch')}")
        with gr.Row():
            with gr.Column():
                self.base_model = gr.Dropdown(label=self._("base_model"), choices=["runwayml/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"], value="runwayml/stable-diffusion-v1-5")
                self.fallback_models = gr.Textbox(label=self._("fallback_models"), value="CompVis/stable-diffusion-v1-4")
            with gr.Column():
                gr.Markdown(f"### {self._('lora_params')}")
                self.lora_rank = gr.Slider(label=self._("lora_rank"), minimum=1, maximum=32, value=4, step=1)
                self.lora_alpha = gr.Slider(label=self._("lora_alpha"), minimum=1, maximum=64, value=32, step=1)
                self.lora_target_modules = gr.Textbox(label=self._("lora_modules"), value="to_q,to_k,to_v,to_out.0")
        
        gr.Markdown(f"### {self._('memory_opt')}")
        with gr.Row():
            self.use_fp16 = gr.Checkbox(label=self._("use_fp16"), value=True)
            self.enable_xformers = gr.Checkbox(label=self._("use_xformers"), value=True)
            self.enable_attention_slicing = gr.Checkbox(label=self._("attention_slicing"), value=True)
            self.enable_vae_slicing = gr.Checkbox(label=self._("vae_slicing"), value=True)
        
        gr.Markdown(f"### {self._('advanced_train')}")
        with gr.Row():
            self.gradient_checkpointing = gr.Checkbox(label=self._("grad_checkpoint"), value=False)
            self.max_grad_norm = gr.Textbox(label=self._("max_grad_norm"), value="1.0")
            self.seed = gr.Textbox(label=self._("random_seed"), value="42")
    
    def _create_control_ui(self):
        gr.Markdown(f"### {self._('control_panel')}")
        with gr.Row():
            with gr.Column(scale=1):
                self.start_btn = gr.Button(self._("start_training"), variant="primary", size="lg")
                self.stop_btn = gr.Button(self._("stop_training"), variant="stop", size="lg")
                self.status_text = gr.Markdown(self._("status_waiting"))
                self.resume_checkpoint = gr.Textbox(label=self._("resume_checkpoint"), placeholder=self._("resume_info"))
            with gr.Column(scale=2):
                self.progress = gr.Progress()
                self.log_output = gr.Textbox(label=self._("training_logs"), lines=15, interactive=False, autoscroll=True)
        
        def start_training(ds_name, out_name, method, epochs, bs, grad_acc, lr, img_size, val_split, base_m, fb, lr_r, lr_a, lr_t, fp16, xf, attn, vae, gc, mg, sd, resume):
            if self.is_training:
                return self._("status_training_running"), ""
            dataset_path = Path(f"./datasets/{ds_name}")
            if not dataset_path.exists():
                return self._("status_dataset_not_found"), ""
            config = {
                "dataset_path": str(dataset_path), "output_path": f"./outputs/{out_name}",
                "logs_path": "./logs", "checkpoints_path": f"./checkpoints/{out_name}",
                "dataset": {"image_size": img_size, "validation_split": val_split},
                "training": {"method": method, "num_epochs": epochs, "batch_size": bs,
                            "gradient_accumulation_steps": grad_acc, "learning_rate": float(lr)},
                "model": {"base_model": base_m, "fallback_models": [fb], "use_fp16": fp16,
                         "enable_xformers": xf, "enable_attention_slicing": attn, "enable_vae_slicing": vae,
                         "gradient_checkpointing": gc}, "export": {"save_checkpoints": True},
                "logging": {"tensorboard": False}
            }
            config_path = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            self.is_training = True
            def thread():
                try:
                    pipeline = AutoFinetunePipeline(config_path)
                    pipeline.run()
                finally:
                    self.is_training = False
                    if os.path.exists(config_path):
                        os.remove(config_path)
            threading.Thread(target=thread).start()
            return self._("status_training_started"), ""
        
        def stop():
            self.is_training = False
            return self._("status_training_stopped"), ""
        
        self.start_btn.click(start_training, [self.dataset_name, self.output_name, self.method, self.num_epochs, self.batch_size, self.gradient_accumulation, self.learning_rate, self.image_size, self.validation_split, self.base_model, self.fallback_models, self.lora_rank, self.lora_alpha, self.lora_target_modules, self.use_fp16, self.enable_xformers, self.enable_attention_slicing, self.enable_vae_slicing, self.gradient_checkpointing, self.max_grad_norm, self.seed, self.resume_checkpoint], [self.status_text, self.log_output])
        self.stop_btn.click(stop, outputs=[self.status_text, self.log_output])
    
    def _create_monitor_ui(self):
        gr.Markdown(f"### {self._('monitor_title')}")
        self.metrics_display = gr.Markdown(f"### {self._('metrics_table')}\n\n{self._('waiting_metrics')}")
        refresh_btn = gr.Button(self._("refresh_metrics"))
        def refresh():
            outputs = Path("./outputs")
            if outputs.exists():
                models = list(outputs.iterdir())
                if models:
                    latest = max(models, key=lambda p: p.stat().st_mtime)
                    loss_file = latest / "loss_history.json"
                    if loss_file.exists():
                        with open(loss_file) as f:
                            loss_history = json.load(f)
                        if loss_history:
                            return f"### {self._('metrics_table')}\n\n| {self._('metric_last_loss')} | {self._('metric_avg_loss')} | {self._('metric_min_loss')} | {self._('metric_total_steps')} |\n|---|---|---|---|\n| {loss_history[-1]:.6f} | {sum(loss_history[-10:])/min(10,len(loss_history)):.6f} | {min(loss_history):.6f} | {len(loss_history)} |"
            return f"### {self._('metrics_table')}\n\n{self._('waiting_metrics')}"
        refresh_btn.click(refresh, outputs=[self.metrics_display])
    
    def _create_inference_ui(self):
        gr.Markdown(f"### {self._('inference_title')}")
        with gr.Row():
            with gr.Column(scale=1):
                self.model_select = gr.Dropdown(label=self._("select_model"), choices=self._get_available_models())
                self.prompt = gr.Textbox(label=self._("prompt"), placeholder=self._("prompt_placeholder"), lines=3)
                self.negative_prompt = gr.Textbox(label=self._("negative_prompt"), placeholder=self._("negative_placeholder"), lines=2)
                with gr.Row():
                    self.num_steps = gr.Slider(label=self._("inference_steps"), minimum=10, maximum=100, value=30, step=5)
                    self.guidance_scale = gr.Slider(label=self._("guidance_scale"), minimum=1, maximum=15, value=7.5, step=0.5)
                generate_btn = gr.Button(self._("generate_btn"), variant="primary")
                refresh_models_btn = gr.Button(self._("refresh_models"))
            with gr.Column(scale=2):
                self.generated_image = gr.Image(label=self._("generated_image"), height=512)
                self.generation_status = gr.Markdown("")
        
        def refresh():
            return gr.Dropdown(choices=self._get_available_models())
        def generate(model, prompt, neg, steps, scale):
            if not model or not prompt:
                return None, self._("status_model_selected" if not model else self._("status_prompt_empty"))
            model_path = Path(f"./outputs/{model}/final_model")
            if not model_path.exists():
                return None, self._("status_model_not_found", path=str(model_path))
            try:
                from diffusers import StableDiffusionPipeline
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pipe = StableDiffusionPipeline.from_pretrained(str(model_path), torch_dtype=torch.float32, safety_checker=None).to(device)
                image = pipe(prompt, negative_prompt=neg if neg else None, num_inference_steps=int(steps), guidance_scale=scale).images[0]
                return image, self._("status_generation_success")
            except Exception as e:
                return None, self._("status_generation_error", error=str(e))
        refresh_models_btn.click(refresh, outputs=[self.model_select])
        generate_btn.click(generate, [self.model_select, self.prompt, self.negative_prompt, self.num_steps, self.guidance_scale], [self.generated_image, self.generation_status])
    
    def _get_available_models(self):
        models = []
        outputs = Path("./outputs")
        if outputs.exists():
            for d in outputs.iterdir():
                if d.is_dir() and (d / "final_model").exists():
                    models.append(d.name)
        return models


def main():
    """Entry point for the application"""
    default_lang = "en"
    if os.path.exists("language_pref.txt"):
        with open("language_pref.txt") as f:
            saved = f.read().strip()
            if saved in ["en", "ru"]:
                default_lang = saved
    gui = PlatformAwareGUI(language=default_lang)
    demo = gui.create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)

if __name__ == "__main__":
    main()