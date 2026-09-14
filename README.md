# SINC

SINC（Spatiotemporal Implicit Neural Fields for Continuous Land-Surface Monitoring）利用多分辨率哈希编码和谐波时间模型，将历史 Sentinel-2 表面反射率序列表示为可连续查询的概率基线。该仓库提供模型训练、训练期终端稳定状态的经验零分布校准、单景或批量异常检测，以及一个可直接运行的 Vegetation Fire 示例。

## 方法与实现边界

1. 输入 GeoTIFF 必须已完成云、云影、雪和无效观测剔除。SINC 不在训练阶段重复执行遥感质量掩膜。
2. 模型从第一轮开始统一使用异方差高斯负对数似然，不使用 L1 预热。
3. 第一幅和最后一幅有效训练观测分别定义归一化时间 `t=0` 和 `t=1`。外推时，趋势项、振幅变化项和动态修正在 `t=1` 后冻结，周期谐波继续演化。
4. 动态分支在训练起点采用零锚定，并接受时间总变差和幅值约束。
5. 预测不确定性随位置、波段、训练期时间和外推距离变化。
6. 经验零分布仅使用 GNDC 训练范围内最新稳定地表状态的残差，不读取目标影像或扰动标签。
7. 二值检测规则为 `D2 > tau_D2 AND (CFAR > tau_CFAR OR SAM > tau_SAM)`。D2、CFAR、SAM 及联合阈值均由经验零分布确定，不设置固定 SAM 下限。

## 目录结构

```text
SINC/
├── sinc/                         # 面向使用者的训练、校准、检测和自检入口
├── ground/                       # GNDC 网络、训练和模型读写
├── onboard/                      # 推理、经验零分布检测和结果导出
├── common/                       # 训练与推理共享函数
├── experiments/adaptive_detection/
│   └── empirical_null_calibration.py
├── configs/                      # 可运行示例配置
├── examples/vegetation_fire/     # 一键示例与外推验证脚本
├── data/Vegetation Fire/         # 106 幅训练影像和 1 幅目标影像
├── scripts/                      # 数据清单生成等维护脚本
├── tests/                        # 不依赖完整训练的单元与完整性测试
├── environment.yml
├── pyproject.toml
└── SOURCE_SYNC_MANIFEST.json
```

## 运行条件

- Python 3.10 或更高版本
- 支持 CUDA 的 NVIDIA GPU
- CUDA 版 PyTorch 2.2 或更高版本
- CUDA Toolkit 和可用的 C++ 编译器
- tiny-cuda-nn 的 PyTorch bindings

完整哈希网格模型依赖 CUDA 和 `tiny-cuda-nn`，不能在仅 CPU 的环境中训练或推理。

## 安装

在仓库根目录执行：

```powershell
conda env create -f environment.yml
conda activate sinc
python -m pip install -e ".[test]"
python -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

`tiny-cuda-nn` 会在本机编译 CUDA 扩展。Windows 用户需要安装与 CUDA 兼容的 Visual Studio C++ Build Tools，并保证 `nvcc` 可用。安装命令与平台要求以 [tiny-cuda-nn 官方说明](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)为准。

如需输出 Shapefile，再安装可选依赖：

```powershell
python -m pip install -e ".[vector]"
```

## 发布包自检

完成安装后运行：

```powershell
python -m sinc.preflight
```

该命令检查 Python 版本、依赖、CUDA、`tiny-cuda-nn`、五波段示例配置、受清单跟踪的发布文件，以及 107 幅示例影像的 SHA-256 和波段布局。仅检查源码和数据完整性时可执行：

```powershell
python -m sinc.preflight --skip-gpu
```

运行不需要完整训练的测试：

```powershell
python -m unittest discover -s tests -v
```

## 示例数据

```text
data/Vegetation Fire/
├── train/   # 模型训练和训练期经验零分布校准
├── target/  # 仅用于最终异常检测
└── DATA_MANIFEST.csv
```

示例 GeoTIFF 保留 7 个源波段，顺序为 `B2, B3, B4, B5, B8, B11, B12`。与论文最终实验一致，训练、校准和检测仅选择其中的 `B3, B4, B8, B11, B12`，即 green、red、NIR、SWIR1 和 SWIR2。该映射在 YAML 中以一基索引 `source_band_indices: [2, 3, 5, 6, 7]` 显式声明，并随 GNDC 模型保存，避免推理时误用前五个源波段。

影像像元为 Sentinel-2 表面反射率数字量化值，配置中的比例因子 `0.0001` 将其转换为反射率。仅保留五个波段均位于 `(0, 1]` 的像元。文件名必须包含 `YYYY-MM-DD` 格式的成像日期；同一天存在多幅源影像时，程序先按像元和波段计算 `nanmedian`，再将其作为一个日观测输入模型。

示例包含 106 幅训练影像和 1 幅后续目标影像。目标影像不参与模型训练、经验阈值估计或 alpha 选择。`DATA_MANIFEST.csv` 记录文件名、日期、尺寸、坐标参考和 SHA-256，可用于检查下载或复制后的数据完整性。

### 与论文实验的对应范围

公开示例复现论文采用的五波段 SINC 训练、基于历史观测的经验零分布校准和 Vegetation Fire 单日期检测流程。论文中的七区域汇总、原生 COLD/S-CCD 对比、区域间 alpha 选择和消融实验所需的其他区域影像与人工参考样本未包含在本仓库中，因此本示例不单独复现这些汇总结果。

## 一键运行

在 `SINC` 根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File examples/vegetation_fire/run_example.ps1
```

脚本依次完成：

1. 使用全部训练影像训练完整 SINC 模型；
2. 从训练期终端稳定状态残差拟合区域经验零分布，并将阈值写入新的 GNDC；
3. 对目标影像执行异常检测。

已有由当前代码训练的模型时，可跳过训练：

```powershell
powershell -ExecutionPolicy Bypass -File examples/vegetation_fire/run_example.ps1 -SkipTraining
```

模型文件包含版本契约。旧版 GNDC 会被当前代码拒绝，不能通过 `-SkipTraining` 与当前校准和检测逻辑混用。

## 分步运行

### 1. 训练概率基线

```powershell
python -m sinc.train --config configs/vegetation_fire.yaml
```

主要输出：

```text
outputs/vegetation_fire/sinc_weights.gndc
```

配置中的 `global_start_date` 和 `global_end_date` 用于筛选训练数据，程序以筛选后第一幅和最后一幅实际训练影像定义时间域。`temporal_buffer_days` 必须为 0。`input_quality_screened: true` 表示使用者确认输入已在上游完成质量控制。

### 2. 校准经验零分布

```powershell
python -m sinc.calibrate `
  --model outputs/vegetation_fire/sinc_weights.gndc `
  --normal-dir "data/Vegetation Fire/train" `
  --output outputs/vegetation_fire/thresholds.json `
  --calibrated-model outputs/vegetation_fire/sinc_calibrated.gndc `
  --alpha 0.005 `
  --device cuda
```

校准程序会：

- 查询每个训练日期的预测均值、不确定性、动态修正及其变化；
- 根据真实观测间隔识别每个像元最近一次状态转换后的稳定后缀；
- 同时要求至少 3 次稳定观测和 30 个稳定日；
- 从终端稳定状态样本估计 D2、CFAR 和 SAM 的经验参考尺度；
- 使用经验联合正常分数的上尾概率确定最终区域阈值；
- 保存阈值 JSON、嵌入阈值的 GNDC 和独立审计 JSON。

`alpha=0.005` 表示经验联合正常分数的 0.5% 上尾概率。该值是论文在七区域 threshold-validation observations 上选择的共同工作点；公开单区域示例直接使用该值，并不在目标扰动影像上重新选择 alpha。日期质量控制不按残差大小删除难例。

### 3. 检测目标影像

```powershell
python -m sinc.detect `
  --model outputs/vegetation_fire/sinc_calibrated.gndc `
  --input "data/Vegetation Fire/target/S2_JZ_2021-09-11_1903.tif" `
  --output outputs/vegetation_fire/detection `
  --device cuda `
  --pixel-size-m 20
```

也可以将 `--input` 指向包含多幅待检测 GeoTIFF 的目录。检测程序根据每幅影像文件名中的日期查询对应的预测均值和不确定性，再应用 GNDC 中锁定的经验零分布配置。

归一化联合分数大于 1 与上述二值判定式严格等价。默认不执行形态学后处理，因此输出掩膜与判定规则逐像元一致。部署任务确需清理小斑块时，可显式增加 `--apply-morphology`，并将其记录为判定后的独立处理。增加 `--export-vector` 可输出矢量结果。

由于五波段流程不包含 B2（blue），检测结果中的三通道浏览底图使用 NIR-red-green 组合，仅用于定位异常，不应解释为真彩色影像。

## 外推验证示例

下列脚本按时间截断训练序列，并评价后续 22 幅事件前影像：

```powershell
powershell -ExecutionPolicy Bypass -File examples/vegetation_fire/run_extrapolation.ps1
```

结果写入：

```text
outputs/vegetation_fire/extrapolation/metrics.csv
```

## 输出文件

```text
outputs/vegetation_fire/
├── sinc_weights.gndc             # 训练得到的概率基线模型
├── sinc_calibrated.gndc          # 嵌入经验零分布阈值的部署模型
├── thresholds.json               # 可独立读取的阈值配置
├── thresholds.audit.json         # 校准日期与状态筛选审计
└── detection/                    # 栅格、JSON和可选矢量检测结果
```

## Python 接口

```python
from sinc import SINCInference

model = SINCInference("outputs/vegetation_fire/sinc_calibrated.gndc", device="cuda")
mean, std = model.predict_frame(date, height, width)
```

命令行入口在执行 `pip install -e .` 后同样可用：

```text
sinc-train
sinc-calibrate
sinc-detect
sinc-check
```

## 代码与数据完整性

`SOURCE_SYNC_MANIFEST.json` 保存公开发布版本中代码、配置、示例入口和测试文件的 SHA-256。`python -m sinc.preflight` 会验证这些文件未被意外修改。示例影像的校验信息保存在 `data/Vegetation Fire/DATA_MANIFEST.csv`。

源码或配置发生有意更新时，应重新生成源码清单：

```powershell
python scripts/build_source_manifest.py
```

若示例数据发生有意更新，应重新生成数据清单：

```powershell
python scripts/build_data_manifest.py
```

## 复现注意事项

- 固定 Python、PyTorch、CUDA 和 `tiny-cuda-nn` 版本。
- 固定 YAML 中的随机种子和训练超参数。
- 保存上游质量掩膜规则及输入影像清单。
- 不要混用旧 GNDC、旧固定阈值或旧检测结果。
- GPU 架构、CUDA 扩展版本和浮点精度可能带来很小的数值差异。
