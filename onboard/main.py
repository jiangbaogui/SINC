"""SINC anomaly-detection command-line entry point.

Usage:
    # 单景检测
    python onboard/main.py --model model.gndc --input image.tif --output ./results
    
    # 批量检测
    python onboard/main.py --model model.gndc --input ./images --output ./results
    
    # 参数配置
    python onboard/main.py --model model.gndc --input image.tif --output ./results \
        --device cuda --chunk-size 262144
"""
import os
import numpy as np
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onboard.core.onboard_inference import OnboardInference
from onboard.core.anomaly_detector import AnomalyDetector
from onboard.preprocessing.image_preprocessor import ImagePreprocessor
from onboard.core.result_exporter import ResultExporter


class OnboardAnomalyDetector:
    """Run date-resolved empirical-null anomaly detection with a GNDC."""
    
    def __init__(
        self,
        model_path,
        threshold_profile=None,
        device=None,
        chunk_size=None,
        pixel_size_m=20.0,
        export_vector=False,
        apply_morphology=False,
    ):
        """
        初始化
        
        Args:
            model_path: .gndc模型文件路径
            device: 设备类型 ('cuda' 或 'cpu')
            chunk_size: 推理分块大小
        """
        print("[System] Initializing SINC anomaly detector...")
        print(f"[System] Loading model: {model_path}")
        
        self.inference = OnboardInference(model_path, device=device)
        self.inference.require_current_model_contract(expected_ablation_mode="full")
        self.meta = self.inference.get_meta()
        self.source_band_indices = self.meta.get("source_band_indices")
        self.source_band_names = self.meta.get("source_band_names")
        threshold_profile = (
            threshold_profile or self.inference.get_empirical_null_profile()
        )
        if threshold_profile is None:
            raise ValueError(
                "No empirical-null profile was supplied or embedded in the GNDC."
            )
        
        band_indices = self.inference.get_band_indices()
        self.detector = AnomalyDetector(
            self.meta, band_indices, threshold_profile=threshold_profile
        )
        
        self.chunk_size = chunk_size
        self.pixel_size_m = float(pixel_size_m)
        if self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive")
        self.pixel_area_m2 = self.pixel_size_m**2
        self.export_vector = bool(export_vector)
        self.apply_morphology = bool(apply_morphology)
        
        print(f"[System] Model ready: {self.meta['n_bands']} bands")
        print(f"[System] Date range: {self.meta['global_start_date']} - {self.meta['global_end_date']}")
        
    def detect_single(self, tif_path, output_dir):
        """
        检测单景影像
        
        Args:
            tif_path: 输入TIF文件路径
            output_dir: 输出目录
        Returns:
            result: 检测结果字典
        """
        print(f"\n[Process] Detecting: {Path(tif_path).name}")
        start_time = time.time()
        
        preprocessor = ImagePreprocessor()
        obs, img_meta = preprocessor.load_tif(
            tif_path,
            expected_bands=self.meta['n_bands'],
            source_band_indices=self.source_band_indices,
            source_band_names=self.source_band_names,
        )
        
        date_dt = img_meta['date']
        if date_dt is None:
            print(f"[Error] No valid date found in filename")
            return None
        
        H, W, C = obs.shape
        
        print(f"[Inference] Running model prediction...")
        pred_mean, pred_std = self.inference.predict_frame(
            date_dt, H, W, chunk_size=self.chunk_size
        )
        
        print(f"[Detection] Running anomaly detection...")
        mask, score, area_km2 = self.detector.detect(
            obs,
            pred_mean,
            pred_std,
            acquisition_date=date_dt,
            pixel_area_m2=self.pixel_area_m2,
            clean=self.apply_morphology,
        )
        
        if area_km2 > 0:
            print(f"[Alert] Anomaly detected! Area: {area_km2:.4f} km²")
            
            transform = img_meta.get('transform')
            crs = img_meta.get('crs')
            
            # ====================================================
            # 🚀 核心修复：将 list 转换为 rasterio 兼容的 Affine 对象
            # ====================================================
            if transform and crs:
                import rasterio
                from rasterio.transform import Affine
                
                if isinstance(transform, list):
                    affine_transform = Affine(*transform)
                else:
                    affine_transform = transform
                    
                detections = self.detector.get_detections(mask, score, affine_transform, crs)
            else:
                detections = []
            # ====================================================
            
            exporter = ResultExporter(
                output_dir, pixel_area_m2=self.pixel_area_m2
            )
            
            tiff_path = exporter.export_visual_geotiff(obs, mask, score, img_meta, area_km2, rgb_bands=(2, 1, 0))
            shp_path = (
                exporter.export_shapefile(detections, img_meta)
                if self.export_vector
                else None
            )
            json_path = exporter.export_json_alert(mask, score, img_meta, detections)
                
            result = {
                    'path': tiff_path,    # 🚀 修复：这里应该是 tiff_path (双f)
                    'shp': shp_path,      # 🚀 新增：把生成的 shp 路径也记录进结果里
                    'json': json_path,
                    'anomaly_detected': True,
                    'area_km2': area_km2,
                    'detection_count': int(np.sum(mask)),
                    'detections': detections
                }
        else:
            print(f"[Info] No anomaly detected.")
            exporter = ResultExporter(
                output_dir, pixel_area_m2=self.pixel_area_m2
            )
            exporter.export_compact_alert(mask, img_meta)
            
            result = {
                'anomaly_detected': False,
                'area_km2': 0,
                'detection_count': 0
            }
        
        elapsed = time.time() - start_time
        print(f"[Complete] Time: {elapsed:.2f}s")
        
        return result
    
    def detect_batch(self, input_path, output_dir, pattern="*.tif"):
        """
        批量检测
        
        Args:
            input_path: 输入目录或文件
            output_dir: 输出目录
            pattern: 文件匹配模式
            
        Returns:
            results: 检测结果列表
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            tif_files = [input_path]
        else:
            tif_files = sorted(input_path.glob(pattern))
        
        print(f"[Batch] Found {len(tif_files)} files to process")
        
        results = []
        for tif_path in tif_files:
            try:
                result = self.detect_single(str(tif_path), output_dir)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[Error] Failed to process {tif_path}: {e}")
                import traceback
                traceback.print_exc()
        
        exporter = ResultExporter(output_dir)
        exporter.export_summary(results)
        
        print(f"\n[Batch Complete] Processed {len(results)} scenes")
        
        anomaly_count = sum(1 for r in results if r.get('anomaly_detected', False))
        print(f"[Summary] Scenes with anomalies: {anomaly_count}/{len(results)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="SINC anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model', '-m', required=True,
                        help='Path to .gndc model file')
    parser.add_argument('--input', '-i', required=True,
                        help='Input TIF file or directory')
    parser.add_argument('--threshold-profile', '-t', default=None,
                        help='Optional empirical-null JSON; otherwise use the profile embedded in GNDC')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory')
    parser.add_argument('--device', '-d', default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--chunk-size', '-c', type=int, default=262144,
                        help='Inference chunk size (default: 262144)')
    parser.add_argument('--pattern', '-p', default='*.tif',
                        help='File pattern for batch processing (default: *.tif)')
    parser.add_argument('--pixel-size-m', type=float, default=20.0,
                        help='Pixel size in metres used for area reporting (default: 20)')
    parser.add_argument('--export-vector', action='store_true',
                        help='Export detected polygons as an ESRI Shapefile')
    parser.add_argument('--apply-morphology', action='store_true',
                        help='Optionally clean the binary mask after the calibrated decision')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SINC Anomaly Detection")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Threshold profile: {args.threshold_profile}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    detector = OnboardAnomalyDetector(
        model_path=args.model,
        threshold_profile=args.threshold_profile,
        device=args.device,
        chunk_size=args.chunk_size,
        pixel_size_m=args.pixel_size_m,
        export_vector=args.export_vector,
        apply_morphology=args.apply_morphology,
    )
    
    input_path = Path(args.input)
    if input_path.is_dir():
        detector.detect_batch(
            input_path=args.input,
            output_dir=args.output,
            pattern=args.pattern
        )
    else:
        detector.detect_single(
            tif_path=args.input,
            output_dir=args.output
        )
    
    print("\n[Done]")


if __name__ == "__main__":
    main()
