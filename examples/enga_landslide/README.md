# HSTB-IR Enga landslide example

This example generates a five-band reflectance baseline for the Sentinel-2 acquisition of 28 May 2024 in Enga Province, Papua New Guinea. It then applies the historical empirical-null screening thresholds and computes metrics on the accompanying interpreted reference samples.

## Run the example

Use Python 3.10 or later with an NVIDIA CUDA GPU, a CUDA-enabled PyTorch installation, and [tiny-cuda-nn PyTorch bindings](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) built for the same environment. Install the remaining dependencies and run the example from the repository root.

```shell
python -m pip install -r examples/enga_landslide/requirements.txt
python examples/enga_landslide/run_case.py --output enga_output
```

The output includes the predicted reflectance, residual scale, disturbance mask and reference-sample metrics. The five output bands are B3, B4, B8, B11 and B12. The reference run scored 46 true positives, 24 false negatives, 2 false positives and 130 true negatives (F1 = 77.97%). The runner checks the target and reference sample file checksums and reports whether the generated mask matches the recorded array checksum.

## Included files

- `data/baseline.sinc` contains the fitted harmonic representation, per-pixel state-correction coefficients and historical residual scales for 78,120 pixels.
- `data/empirical_null.json` contains the precomputed calibration profile.
- `data/sentinel2_2024-05-28.tif` contains the processed target acquisition. The original Sentinel-2 imagery is available through the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
- `data/reference_samples.csv` contains the interpreted evaluation points.
- `reference/` contains the original experiment protocol and metrics. The protocol lists the dates used for fitting.

The example starts from a frozen model and calibration profile. Rebuilding those artifacts requires the historical image stack, which is separate from this example. The local Python modules are a snapshot of the inference implementation used for this case. The repository's earlier Vegetation Fire example uses the original SINC workflow.
