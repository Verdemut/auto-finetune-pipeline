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
        self.current_platform = "local"
        
        # Language dictionaries
        self.strings = {
            "en": {
                "title": "Auto-Finetune Pipeline",
                "subtitle": "Professional Neural Network Training Interface",
                "platform_select": "Training Platform",
                "platform_local": "Local PC (GPU/CPU)",
                "tab_dataset": "Dataset Management",
                "tab_config": "Training Configuration",
                "tab_advanced": "Advanced Settings",
                "tab_control": "Training Control",
                "tab_monitor": "Monitoring",
                "tab_inference": "Inference",
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
                "restart_button": "Apply and Restart",
            },
            "ru": {
                "title": "Auto-Finetune Pipeline",
                "subtitle": "Профессиональный интерфейс обучения нейросетей",
                "platform_select": "Платформа для обучения",
                "platform_local": "Локальный ПК (GPU/CPU)",
                "tab_dataset": "Управление датасетом",
                "tab_config": "Настройки обучения",
                "tab_advanced": "Расширенные настройки",
                "tab_control": "Управление обучением",
                "tab_monitor": "Мониторинг",
                "tab_inference": "Генерация",
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
            with gr.Row():
                with gr.Column(scale=8):
                    gr.Markdown(f"# {self._('title')}")
                    gr.Markdown(f"### {self._('subtitle')}")
                with gr.Column(scale=2):
                    lang_selector = gr.Dropdown(
                        choices=[self._("lang_en"), self._("lang_ru")],
                        value=self._("lang_en") if self.language == "en" else self._("lang_ru"),
                        label=self._("language")
                    )
                    restart_btn = gr.Button(self._("restart_button"), variant="secondary", size="sm")
            
            with gr.Row():
                with gr.Column():
                    status_display = gr.Markdown(self._get_hardware_status())
            
            with gr.Tabs():
                with gr.TabItem(self._("tab_dataset")):
                    self._create_dataset_ui()
                
                with gr.TabItem(self._("tab_config")):
                    self._create_training_ui()
                
                with gr.TabItem(self._("tab_advanced")):
                    self._create_advanced_ui()
                
                with gr.TabItem(self._("tab_control")):
                    self._create_control_ui()
                
                with gr.TabItem(self._("tab_monitor")):
                    self._create_monitor_ui()
                
                with gr.TabItem(self._("tab_inference")):
                    self._create_inference_ui()
            
            def restart_app(lang_selected):
                new_lang = "ru" if lang_selected == self._("lang_ru") else "en"
                with open("language_pref.txt", "w") as f:
                    f.write(new_lang)
                python = sys.executable
                subprocess.Popen([python, __file__])
                os._exit(0)
            
            restart_btn.click(restart_app, inputs=[lang_selector], outputs=[])
        
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
                    placeholder="Enter caption",
                    lines=2
                )
                
                with gr.Row():
                    apply_caption_btn = gr.Button(self._("apply_caption"), variant="secondary")
                    save_dataset_btn = gr.Button(self._("save_dataset"), variant="primary")
                
                self.dataset_name = gr.Textbox(label=self._("dataset_name"), value="my_dataset")
                self.dataset_status = gr.Markdown(self._("dataset_status"))
            
            with gr.Column(scale=2):
                self.gallery = gr.Gallery(label=self._("image_preview"), columns=4, rows=2, height=400)
                gr.Markdown(f"### {self._('caption_editor')}")
                self.caption_editor = gr.Dataframe(
                    headers=[self._("filename"), self._("caption")],
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
                self.resume_checkpoint = gr.Textbox(label=self._("resume_checkpoint"), placeholder="Path to checkpoint")
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
                            return f"### {self._('metrics_table')}\n\nLast Loss: {loss_history[-1]:.6f}"
            return f"### {self._('metrics_table')}\n\n{self._('waiting_metrics')}"
        
        refresh_btn.click(refresh, outputs=[self.metrics_display])
    
    def _create_inference_ui(self):
        gr.Markdown(f"### {self._('inference_title')}")
        with gr.Row():
            with gr.Column(scale=1):
                self.model_select = gr.Dropdown(label=self._("select_model"), choices=self._get_available_models())
                self.prompt = gr.Textbox(label=self._("prompt"), placeholder=self._("prompt_placeholder"), lines=3)
                self.negative_prompt = gr.Textbox(label=self._("negative_prompt"), placeholder="", lines=2)
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
                return None, f"Model not found: {model_path}"
            try:
                from diffusers import StableDiffusionPipeline
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pipe = StableDiffusionPipeline.from_pretrained(str(model_path), torch_dtype=torch.float32, safety_checker=None).to(device)
                image = pipe(prompt, negative_prompt=neg if neg else None, num_inference_steps=int(steps), guidance_scale=scale).images[0]
                return image, self._("status_generation_success")
            except Exception as e:
                return None, f"Error: {e}"
        
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