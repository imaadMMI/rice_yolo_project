# Results - yolov8s-cls, imgsz 512, 5 runs (mean ± std)

| Experiment | Scenario | n test | Accuracy (%) | Macro-F1 (%) | Macro-Precision (%) | Macro-Recall (%) |
|---|---|---|---|---|---|---|
| combined | a_white | 154 | 81.17 ± 0.41 | 78.85 ± 0.72 | 80.60 ± 1.37 | 78.28 ± 0.60 |
|  |  |  | Brown_Spot: 68.9, Leaf_Scald: 67.9, Rice_Blast: 86.0, Rice_Tungro: 75.0, Sheath_Blight: 93.6 | | | |
| combined | b_field | 68 | 70.00 ± 7.00 | 68.92 ± 7.81 | 71.46 ± 8.43 | 69.98 ± 6.97 |
|  |  |  | Brown_Spot: 62.0, Leaf_Scald: 44.0, Rice_Blast: 73.3, Rice_Tungro: 81.3, Sheath_Blight: 89.2 | | | |
| combined | c_both | 222 | 77.75 ± 2.38 | 75.91 ± 2.79 | 77.38 ± 3.44 | 75.72 ± 2.58 |
|  |  |  | Brown_Spot: 66.4, Leaf_Scald: 59.5, Rice_Blast: 82.5, Rice_Tungro: 77.4, Sheath_Blight: 92.6 | | | |
| white_only | a_white | 154 | 81.04 ± 2.11 | 78.65 ± 2.26 | 80.05 ± 3.00 | 77.93 ± 1.97 |
|  |  |  | Brown_Spot: 70.0, Leaf_Scald: 70.0, Rice_Blast: 85.5, Rice_Tungro: 69.2, Sheath_Blight: 95.0 | | | |
| white_only | b_field_xdomain | 68 | 34.41 ± 4.80 | 27.03 ± 7.84 | 40.95 ± 10.15 | 33.11 ± 5.26 |
|  |  |  | Brown_Spot: 4.0, Leaf_Scald: 62.7, Rice_Blast: 30.7, Rice_Tungro: 6.7, Sheath_Blight: 61.5 | | | |
| field_only | b_field | 68 | 63.24 ± 4.26 | 61.66 ± 5.73 | 63.10 ± 6.34 | 62.91 ± 5.40 |
|  |  |  | Brown_Spot: 48.0, Leaf_Scald: 38.7, Rice_Blast: 76.0, Rice_Tungro: 62.7, Sheath_Blight: 89.2 | | | |
| field_only | a_white_xdomain | 154 | 32.21 ± 8.13 | 23.75 ± 4.86 | 41.19 ± 13.21 | 31.58 ± 5.94 |
|  |  |  | Brown_Spot: 7.8, Leaf_Scald: 14.3, Rice_Blast: 10.0, Rice_Tungro: 75.8, Sheath_Blight: 50.0 | | | |

Mean training time per model: 16.2 min; inference 12.0 ms/image (incl. JPEG decode).