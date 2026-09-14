"""
配置管理模块 - 解析和管理YAML配置文件
"""
import os
import yaml
from datetime import datetime, date


class NDCConfig:
    """NOMAD配置管理器"""
    
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"Loading Configuration from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self._cfg = yaml.safe_load(f)
        
        self._parse_project()
        self._parse_paths()
        self._parse_data()
        self._parse_train()
        self._parse_output()
    
    def _parse_project(self):
        proj = self._cfg.get('project', {})
        self.project_name = proj.get('name', 'Untitled')
        self.project_description = proj.get('description', '')
    
    def _parse_paths(self):
        paths = self._cfg.get('paths', {})
        self.input_folder = paths.get('input_folder', '')
        self.output_model = paths.get('output_model', '')
        configured_cache = paths.get('cache_dir')
        self.cache_dir = (
            os.path.abspath(os.path.expanduser(str(configured_cache)))
            if configured_cache
            else None
        )
        
        if self.output_model:
            output_dir = os.path.dirname(self.output_model)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
    
    def _parse_data(self):
        data = self._cfg.get('data', {})
        self.dataset_mode = data.get('dataset_mode', 'standard')
        self.n_bands = int(data.get('n_bands', 1))
        configured_band_indices = data.get('source_band_indices')
        if configured_band_indices is None:
            self.source_band_indices = None
        else:
            self.source_band_indices = [int(value) for value in configured_band_indices]
            if len(self.source_band_indices) != self.n_bands:
                raise ValueError(
                    "data.source_band_indices must contain exactly data.n_bands "
                    "one-based GeoTIFF band indices."
                )
            if any(value < 1 for value in self.source_band_indices):
                raise ValueError(
                    "data.source_band_indices must be one-based positive integers."
                )
            if len(set(self.source_band_indices)) != len(self.source_band_indices):
                raise ValueError("data.source_band_indices cannot contain duplicates.")

        configured_band_names = data.get('source_band_names')
        if configured_band_names is None:
            self.source_band_names = None
        else:
            self.source_band_names = [str(value) for value in configured_band_names]
            if len(self.source_band_names) != self.n_bands:
                raise ValueError(
                    "data.source_band_names must contain exactly data.n_bands names."
                )
        
        self.global_start_date = self._parse_date(data.get('global_start_date'))
        self.global_end_date = self._parse_date(data.get('global_end_date'))
        
        self.data_scale = data.get('data_scale', 1.0)
        self.input_valid_range = data.get('input_valid_range', [0.0, 1.0])
        self.input_p99_removal = data.get('input_p99_removal', False)
        self.temporal_bin_days = data.get('temporal_bin_days', None)
        # The temporal domain ends at the final observed training acquisition.
        self.temporal_buffer_days = int(data.get('temporal_buffer_days', 0))
        if self.temporal_buffer_days != 0:
            raise ValueError(
                "temporal_buffer_days is no longer supported; set it to 0. "
                "The trainer derives the endpoint from the last acquisition."
            )
        self.input_quality_screened = bool(
            data.get('input_quality_screened', False)
        )
        self.quality_mask_description = str(
            data.get(
                'quality_mask_description',
                'Cloud, cloud-shadow, and snow masking completed during preprocessing.',
            )
        )
    
    def _parse_train(self):
        train = self._cfg.get('train', {})
        self.epochs = train.get('epochs', 500)
        self.lr = train.get('lr', 0.01)
        self.use_log = train.get('use_log', False)
        self.train_dtype = train.get('train_dtype', 'float16')
        configured_loss = str(train.get('train_loss_type', 'NLL')).upper()
        if configured_loss != 'NLL':
            raise ValueError(
                "SINC training uses heteroscedastic NLL throughout; "
                f"train_loss_type={configured_loss!r} is not supported."
            )
        self.train_loss_type = 'NLL'
        
        self.n_harmonics = train.get('n_harmonics', 3)
        self.lambda_reg = train.get('lambda_reg', 0.005)
        self.lambda_dynamic_magnitude = train.get(
            'lambda_dynamic_magnitude', self.lambda_reg
        )
        self.valid_threshold = train.get('valid_threshold', 0.0)
        
        self.xy_hash_size = train.get('xy_hash_size', 19)
        self.t_hash_size = train.get('t_hash_size', 17)
        self.n_levels = train.get('n_levels', 16)
        self.per_level_scale = train.get('per_level_scale', 1.18)
        self.base_resolution = train.get('base_resolution', 32)
        self.features_per_level = train.get('features_per_level', 2)
        
        self.decoder_hidden = train.get('decoder_hidden', 128)
        self.n_hidden_layers = train.get('n_hidden_layers', 2)
        self.ablation_mode = train.get('ablation_mode', 'full')
        self.random_seed = int(train.get('random_seed', 42))
    
    def _parse_output(self):
        output = self._cfg.get('output', {})
        self.mask_mode = output.get('mask_mode', 'dynamic-xor')
        self.out_dtype = output.get('out_dtype', 'float32')
        
        # 新增：量化模式，可选 'float32', 'float16', 'int16', 'int8'
        self.quant_mode = output.get('quant_mode', 'int16') 
        
        self.save_residuals = output.get('save_residuals', True)
        self.residual_threshold = output.get('residual_threshold', 0.0005)
        self.residual_dtype = output.get('residual_dtype', 'int16')
    def _parse_date(self, date_val):
        if date_val is None:
            return None
        
        if isinstance(date_val, datetime):
            return date_val
        
        if isinstance(date_val, date):
            return datetime(date_val.year, date_val.month, date_val.day)
        
        date_str = str(date_val).strip()
        try:
            if len(date_str) == 10:
                return datetime.strptime(date_str, "%Y-%m-%d")
            else:
                return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"[Warning] Failed to parse date string: {date_str}")
            return None
    
    def get_output_dir(self):
        if self.output_model:
            return os.path.dirname(self.output_model)
        return None
    
    def get_model_name(self):
        if self.output_model:
            return os.path.basename(self.output_model)
        return None


def load_config(config_path: str) -> NDCConfig:
    """便捷函数：加载配置"""
    return NDCConfig(config_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cfg = NDCConfig(sys.argv[1])
        print(f"Project: {cfg.project_name}")
        print(f"Bands: {cfg.n_bands}")
        print(f"Epochs: {cfg.epochs}")
