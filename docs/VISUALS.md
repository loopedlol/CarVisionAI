# CarVisionAI / Visual assets

[← Project](../README.md) · [Technical guide](TECHNICAL_GUIDE.md)

## Asset inventory

- `assets/portfolio/hero-{light,dark}.svg`: original route, occupancy, and field-of-view motif; a schematic, not a measured map.
- `assets/portfolio/system-{light,dark}.svg`: implementation-oriented module map with simulated feedback.
- `assets/portfolio/active-perception/01_rough_candidates.png` through `04_replanned.png`: unedited output from the existing demo.
- `assets/portfolio/social-preview.svg` and `social-preview.png`: 1280 × 640 social card; PNG ready for manual upload.

## Reproduce the demo images

Captured on 2026-09-19 from source commit `1ae574bb1eefdcc445d9ed3a4fa36f9b5a56f2f0`, using Python 3.14.6, NumPy 2.5.3, and OpenCV 5.0.0. Run from the repository root:

```bash
PYTHONPATH=src python scripts/run_active_perception_demo.py --output-dir outputs/planning_demo
```

The unchanged default synthetic scenario produced five initial candidates, five replanned candidates, and 35 changed cells. The selected route changed. These describe this one demonstration, not a benchmark or hardware result. The four PNGs were retained without altering their map or annotations.

## Future real-world evidence

A calibrated stereo pair and a photo of the actual robot would add context. No placeholder hardware photo is shown, and no build is implied by the schematic. Publish them only with verified hardware details and permission for anything visible in the frame.
