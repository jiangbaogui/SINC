"""
结果输出模块 - 导出出版级GeoTIFF（带RGB底图、热力图、标题和色带图例）
=============================
Author: NOMAD-01 Onboard System
Modified to support professional PDF-like visualization export to GeoTIFF.
"""
import os
import json
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import geopandas as gpd
    from shapely.geometry import shape
except ImportError:
    print("[Warning] geopandas or shapely not found. SHP export may fail.")

try:
    from common.constants import ALERT_VERSION, OUTPUT_DTYPE
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from common.constants import ALERT_VERSION, OUTPUT_DTYPE


class ResultExporter:
    def __init__(self, output_dir, satellite_id="SINC", pixel_area_m2=400.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.satellite_id = satellite_id
        self.pixel_area_m2 = float(pixel_area_m2)
        if self.pixel_area_m2 <= 0:
            raise ValueError("pixel_area_m2 must be positive")
        
    def export_visual_geotiff(self, obs, mask, score, meta, area_km2, rgb_bands=(2, 1, 0), prefix="anomaly_visual"):
        """
        🚀 [紧凑极限版]: 极大减少留白，提升图像占比，保持字体比例协调。
        """
        import numpy as np
        import rasterio
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import Normalize

        H, W = mask.shape
        transform = meta.get('transform')
        crs = meta.get('crs')
        date_str = meta.get('date_str', 'unknown')
        
        # 1. 底图拉伸逻辑 (保持不动)
        rgb_base = np.zeros((H, W, 3), dtype=np.float32)
        for i, b_idx in enumerate(rgb_bands):
            band = obs[:, :, b_idx]
            valid_pixels = band[band > 0]
            if len(valid_pixels) > 0:
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                stretched = np.clip((band - p2) / (p98 - p2 + 1e-9), 0, 1)
                rgb_base[:, :, i] = stretched
        rgb_base = (rgb_base * 255).astype(np.uint8)

        # ------------------- 布局参数极致收缩 -------------------
        dpi = 100
        TITLE_FONT_SIZE = 12 
        LEGEND_FONT_SIZE = 10 
        
        # 🚨 [核心修改]: 显著缩小画布尺寸冗余 (W/dpi + 2.0, H/dpi + 1.8)
        fig = Figure(figsize=(W / dpi + 2.2, H / dpi + 2.0), dpi=dpi, facecolor='black') 
        canvas = FigureCanvasAgg(fig)
        
        # 🚨 [核心修改]: 极大扩展主图坐标轴占比
        # ax_img: [左, 下, 宽, 高] -> 图像占据 82% 的宽度和 78% 的高度
        # 将左边距收缩到 0.04，底边距保持 0.12 以放下图例文字
        ax_img = fig.add_axes([0.04, 0.12, 0.82, 0.78], facecolor='black')
        
        # ax_cb: 图例渐变条位置紧贴图像右侧 (0.88位置)
        ax_cb = fig.add_axes([0.88, 0.12, 0.02, 0.78]) 
        
        # 显示内容
        ax_img.imshow(rgb_base, origin='upper', interpolation='nearest')
        score_masked = np.where(mask, score, np.nan)
        norm = Normalize(vmin=1.0, vmax=3.0)
        im_ov = ax_img.imshow(score_masked, cmap='inferno', norm=norm, origin='upper', alpha=0.9) 
        ax_img.axis('off') 

        # 2. 标题：垂直位置下移，贴近图像
        title_str = f"{date_str}  |  Area: {area_km2:.4f} km²"
        fig.text(0.45, 0.94, title_str, color='white', 
                 fontsize=TITLE_FONT_SIZE, fontweight='bold', ha='center')

        # 3. 图例：数字显示，标签在正下方
        from matplotlib.colorbar import Colorbar
        cb = Colorbar(ax_cb, im_ov)
        cb.outline.set_edgecolor('#666666')
        
        ticks = [1.0, 1.5, 2.0, 2.5, 3.0]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}" for t in ticks])
        
        cb.ax.tick_params(labelsize=LEGEND_FONT_SIZE, colors='white', length=3, direction='in')
        
        # 标签放在色带正下方
        ax_cb.set_xlabel('Anomaly\nSeverity', color='white', fontsize=LEGEND_FONT_SIZE, 
                         labelpad=8, ha='center')

        # ------------------- 坐标对齐逻辑 (确保 GIS 精度) -------------------
        canvas.draw()
        rgba_canvas = np.asarray(canvas.buffer_rgba()) 
        rgb_canvas = rgba_canvas[..., :3] 

        bbox_img = ax_img.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        axes_extent = [bbox_img.x0/fig.get_size_inches()[0], bbox_img.y0/fig.get_size_inches()[1], 
                       bbox_img.width/fig.get_size_inches()[0], bbox_img.height/fig.get_size_inches()[1]]
        
        H_new, W_new, _ = rgb_canvas.shape
        a, b, c, d, e, f = transform[:6]
        
        x_shift = axes_extent[0] * fig.get_size_inches()[0] * fig.dpi
        y_shift = (1.0 - axes_extent[1] - axes_extent[3]) * fig.get_size_inches()[1] * fig.dpi
        x_scale = (fig.get_size_inches()[0] * fig.dpi) / W
        y_scale = (fig.get_size_inches()[1] * fig.dpi) / H
        
        new_c = c - (x_shift * a / x_scale)
        new_f = f - (y_shift * e / y_scale)
        new_a = a / x_scale
        new_e = e / y_scale
        new_affine = rasterio.transform.Affine(new_a, b, new_c, d, new_e, new_f)

        profile = {
            'driver': 'GTiff', 'height': H_new, 'width': W_new, 'count': 3,
            'dtype': 'uint8', 'transform': new_affine, 'compress': 'lzw', 'photometric': 'RGB'
        }
        if crs: profile['crs'] = crs

        output_path = self.output_dir / f"{prefix}_{date_str}_compact.tif"
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(3):
                dst.write(rgb_canvas[:, :, i], i + 1)
            
        return str(output_path)

    def export_shapefile(
        self, detections, meta, prefix="anomaly", write_empty=False
    ):
        if not detections:
            if not write_empty:
                print("[Export] No detections to save to SHP.")
                return None

            import fiona

            date_str = meta.get('date_str', 'unknown')
            shp_path = self.output_dir / f"{prefix}_{date_str}.shp"
            schema = {
                'geometry': 'Polygon',
                'properties': {'area_m2': 'float'},
            }
            crs = meta.get('crs')
            open_options = {
                'driver': 'ESRI Shapefile',
                'schema': schema,
                'encoding': 'utf-8',
            }
            if crs:
                open_options['crs_wkt'] = crs.to_wkt()
            with fiona.open(shp_path, mode='w', **open_options):
                pass
            print(f"[Export] Empty shapefile saved: {shp_path}")
            return str(shp_path)
        geometries = [shape(d['geometry']) for d in detections]
        properties = [{'area_m2': d['area_m2']} for d in detections]
        gdf = gpd.GeoDataFrame(properties, geometry=geometries)
        crs = meta.get('crs')
        if crs: gdf.set_crs(crs, inplace=True, allow_override=True)
        date_str = meta.get('date_str', 'unknown')
        shp_path = self.output_dir / f"{prefix}_{date_str}.shp"
        gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
        print(f"[Export] Shapefile saved: {shp_path}")
        return str(shp_path)
    
    def export_json_alert(self, mask, score, meta, detections=None):
        date_str = meta.get('date_str', datetime.now().strftime("%Y-%m-%d"))
        area_km2 = self._calculate_area(mask)
        detection_count = int(np.sum(mask))
        if detections is None: detections = []
        alert = {
            "version": ALERT_VERSION,
            "satellite": self.satellite_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "scene_id": f"{meta.get('date_str', 'unknown')}",
            "image_date": date_str,
            "anomaly_detected": bool(detection_count > 0),
            "total_area_km2": round(area_km2, 4),
            "detection_count": detection_count,
            "detections": detections,
            "metadata": {
                "image_path": str(meta.get('path', '')),
                "transform": meta.get('transform', []),
                "crs": meta.get('crs', '')
            }
        }
        output_path = self.output_dir / f"alert_{date_str}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alert, f, indent=2, ensure_ascii=False)
        return str(output_path)

    def export_compact_alert(self, mask, meta):
        date_str = meta.get('date_str', datetime.now().strftime("%Y-%m-%d"))
        area_km2 = self._calculate_area(mask)
        alert = {
            "timestamp": datetime.now().isoformat() + "Z",
            "scene_id": date_str,
            "anomaly_detected": bool(np.any(mask)),
            "total_area_km2": round(area_km2, 4)
        }
        output_path = self.output_dir / f"alert_{date_str}_compact.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alert, f)
        return str(output_path)

    def _calculate_area(self, mask):
        return (np.sum(mask) * self.pixel_area_m2) / 1e6

    def export_summary(self, results):
        summary = {
            "generated_at": datetime.now().isoformat() + "Z",
            "satellite": self.satellite_id,
            "total_scenes": len(results),
            "scenes_with_anomalies": sum(1 for r in results if r.get('anomaly_detected', False)),
            "results": results
        }
        output_path = self.output_dir / "summary.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return str(output_path)
