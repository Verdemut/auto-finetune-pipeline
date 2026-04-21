# modules/trainer.py - Без эмодзи
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import os
from pathlib import Path
import traceback


class Trainer:
    def __init__(self, config, hyperparams, logger):
        self.config = config
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Настройка mixed precision
        use_fp16 = hyperparams.get('use_fp16', False) and torch.cuda.is_available()
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=hyperparams['gradient_accumulation_steps'],
            mixed_precision="fp16" if use_fp16 else "no"
        )
        
        self.loss_history = []
        
    def train(self, dataset):
        """Основной цикл обучения"""
        self.logger.info("Starting training...")
        
        device = self.accelerator.device
        self.logger.info(f"Device: {device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"VRAM: {vram:.1f} GB")
        
        # Загрузка модели с fallback
        pipe = self._load_model_with_fallback(device)
        
        # Подготовка компонентов
        unet = pipe.unet
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        vae = pipe.vae
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.config['model']['base_model'], 
            subfolder="scheduler"
        )
        
        # Заморозка
        self._freeze_components(text_encoder, vae)
        
        # Оптимизатор
        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=self.hyperparams['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        # DataLoader
        from torch.utils.data import DataLoader
        train_dataloader = DataLoader(
            dataset['train'],
            batch_size=self.hyperparams['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Learning rate scheduler
        num_training_steps = len(train_dataloader) * self.hyperparams['num_epochs']
        lr_scheduler = get_scheduler(
            self.config['training'].get('lr_scheduler', 'cosine'),
            optimizer=optimizer,
            num_warmup_steps=self.config['training'].get('warmup_steps', 100),
            num_training_steps=num_training_steps
        )
        
        # Подготовка с accelerator
        unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        
        # Обучение
        global_step = 0
        num_epochs = self.hyperparams['num_epochs']
        
        self.logger.info(f"\nTRAINING PARAMETERS:")
        self.logger.info(f"  Epochs: {num_epochs}")
        self.logger.info(f"  Batch size: {self.hyperparams['batch_size']}")
        self.logger.info(f"  Gradient accumulation: {self.hyperparams['gradient_accumulation_steps']}")
        self.logger.info(f"  Learning rate: {self.hyperparams['learning_rate']}")
        self.logger.info(f"  Method: {self.hyperparams['method']}")
        
        try:
            for epoch in range(num_epochs):
                self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
                
                unet.train()
                epoch_loss = 0
                
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    with self.accelerator.accumulate(unet):
                        pixel_values = batch["pixel_values"].to(device)
                        
                        # Кодируем текст
                        text_inputs = tokenizer(
                            batch['caption'],
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        with torch.no_grad():
                            encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
                            latents = vae.encode(pixel_values).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor
                        
                        # Добавляем шум
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (latents.shape[0],), device=device
                        ).long()
                        
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        
                        # Предсказываем шум
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        
                        # Loss
                        loss = torch.nn.functional.mse_loss(noise_pred, noise)
                        
                        # Backward
                        self.accelerator.backward(loss)
                        
                        # Clip gradients
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                unet.parameters(), 
                                self.config['training'].get('max_grad_norm', 1.0)
                            )
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        
                        epoch_loss += loss.item()
                        global_step += 1
                        self.loss_history.append(loss.item())
                        
                        progress_bar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                        })
                
                avg_loss = epoch_loss / len(train_dataloader)
                self.logger.info(f"Epoch {epoch+1} completed | Average loss: {avg_loss:.4f}")
                
                # Сохраняем чекпоинт
                if self.config['export']['save_checkpoints']:
                    if (epoch + 1) % self.config['export']['save_checkpoints_every'] == 0:
                        self._save_checkpoint(unet, epoch, optimizer, global_step)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_checkpoint(unet, epoch, optimizer, global_step, is_final=True)
        
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            traceback.print_exc()
            raise
        
        self.logger.info("Training completed successfully")
        
        return {"unet": unet, "pipe": pipe}
    
    def _load_model_with_fallback(self, device):
        """Загрузка модели с несколькими попытками"""
        models_to_try = [self.config['model']['base_model']]
        models_to_try.extend(self.config['model'].get('fallback_models', []))
        
        dtype = torch.float32  # всегда float32 для стабильности
        
        for model_id in models_to_try:
            try:
                self.logger.info(f"Attempting to load model: {model_id}")
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=False
                )
                pipe = pipe.to(device)
                
                # Оптимизации
                if torch.cuda.is_available():
                    if self.hyperparams.get('use_xformers', False):
                        try:
                            pipe.enable_xformers_memory_efficient_attention()
                            self.logger.info("xFormers enabled")
                        except:
                            self.logger.warning("xFormers not available")
                    
                    if self.config['model'].get('enable_attention_slicing', True):
                        pipe.enable_attention_slicing()
                        self.logger.info("Attention slicing enabled")
                    
                    if self.config['model'].get('enable_vae_slicing', True):
                        pipe.enable_vae_slicing()
                        self.logger.info("VAE slicing enabled")
                
                self.logger.info(f"Model loaded successfully: {model_id}")
                return pipe
                
            except Exception as e:
                self.logger.warning(f"Failed to load {model_id}: {e}")
                continue
        
        raise RuntimeError("Failed to load any model")
    
    def _freeze_components(self, text_encoder, vae):
        """Замораживаем компоненты, которые не обучаем"""
        for param in text_encoder.parameters():
            param.requires_grad = False
        
        for param in vae.parameters():
            param.requires_grad = False
        
        self.logger.info("Text encoder and VAE frozen")
    
    def _save_checkpoint(self, unet, epoch, optimizer, step, is_final=False):
        """Сохранение чекпоинта"""
        if is_final:
            checkpoint_path = Path(self.config['output_path']) / "final_checkpoint"
        else:
            checkpoint_path = Path(self.config['checkpoints_path']) / f"checkpoint_epoch_{epoch+1}"
        
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем UNet
        self.accelerator.unwrap_model(unet).save_pretrained(checkpoint_path)
        
        # Сохраняем состояние
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }, checkpoint_path / "training_state.pt")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")