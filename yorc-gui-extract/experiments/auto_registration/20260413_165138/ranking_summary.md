# Ranking Summary

Score order: face_rmse, full_rmse, -fitness, icp_rmse

## Method Ranking

| Rank | Method | Init | Face RMSE | Full RMSE | Fitness | ICP RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | anterior_crop | ras_to_head_rot_x_-90 | 3.436 | 6.060 | 0.5927 | 1.7417 |
| 2 | face_weighted | ras_to_head_rot_x_-90 | 3.485 | 6.055 | 0.5175 | 1.7567 |
| 3 | full | ras_to_head_rot_x_-90 | 3.490 | 6.059 | 0.3867 | 1.7755 |

## full Candidate Ranking

| Rank | Init | Face RMSE | Full RMSE | Fitness | ICP RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | ras_to_head_rot_x_-90 | 3.490 | 6.059 | 0.3867 | 1.7755 |
| 2 | ras_to_head_rot_x_90 | 6.885 | 8.733 | 0.2036 | 1.7630 |
| 3 | global | 8.576 | 13.129 | 0.1608 | 1.7724 |
| 4 | ras_to_head_rot_y_90_then_rot_z_-90 | 9.006 | 33.230 | 0.1510 | 1.7997 |
| 5 | ras_to_head_rot_x_-90_then_rot_y_90 | 11.481 | 33.096 | 0.0971 | 1.7889 |
| 6 | ras_to_head_rot_x_90_then_rot_y_-90 | 11.819 | 28.938 | 0.0779 | 1.7835 |
| 7 | ras_to_head_rot_x_-90_then_rot_y_-90 | 13.035 | 27.824 | 0.0878 | 1.7721 |
| 8 | ras_to_head | 15.602 | 30.192 | 0.0687 | 1.7917 |
| 9 | ras_to_head_rot_x_90_then_rot_x_90 | 16.208 | 32.929 | 0.0795 | 1.7929 |
| 10 | pca_flip_yz | 17.691 | 17.852 | 0.1522 | 1.7834 |
| 11 | ras_to_head_rot_z_90 | 18.050 | 34.705 | 0.0901 | 1.7880 |
| 12 | ras_to_head_rot_z_-90 | 21.834 | 33.556 | 0.0691 | 1.7990 |
| 13 | ras_to_head_rot_y_-90 | 24.711 | 28.747 | 0.0520 | 1.8589 |
| 14 | pca_flip_xz | 25.242 | 26.749 | 0.0612 | 1.7472 |
| 15 | ras_to_head_rot_x_-90_then_rot_z_90 | 30.503 | 28.340 | 0.0346 | 1.7824 |
| 16 | ras_to_head_rot_x_90_then_rot_z_-90 | 32.079 | 40.219 | 0.0675 | 1.8080 |
| 17 | ras_to_head_rot_y_90 | 32.724 | 55.290 | 0.0599 | 1.7684 |
| 18 | ras_to_head_rot_x_-90_then_rot_z_-90 | 34.560 | 47.796 | 0.0673 | 1.7501 |
| 19 | ras_to_head_rot_y_90_then_rot_y_90 | 36.083 | 61.563 | 0.0465 | 1.7358 |
| 20 | pca_flip_xy | 39.161 | 30.709 | 0.0468 | 1.8046 |
| 21 | pca_identity | 39.347 | 31.698 | 0.0399 | 1.8401 |
| 22 | ras_to_head_rot_z_90_then_rot_z_90 | 40.762 | 65.893 | 0.0291 | 1.6468 |
| 23 | ras_to_head_rot_x_90_then_rot_z_90 | 84.237 | 50.638 | 0.0270 | 1.8204 |

## anterior_crop Candidate Ranking

| Rank | Init | Face RMSE | Full RMSE | Fitness | ICP RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | ras_to_head_rot_x_-90 | 3.436 | 6.060 | 0.5927 | 1.7417 |
| 2 | global | 3.452 | 6.644 | 0.5670 | 1.7345 |
| 3 | ras_to_head_rot_x_90 | 6.820 | 8.721 | 0.2916 | 1.7650 |
| 4 | ras_to_head_rot_x_-90_then_rot_y_-90 | 8.641 | 30.687 | 0.2453 | 1.7003 |
| 5 | ras_to_head_rot_y_90_then_rot_z_-90 | 8.962 | 33.259 | 0.3223 | 1.7804 |
| 6 | ras_to_head_rot_x_-90_then_rot_y_90 | 9.226 | 32.925 | 0.2670 | 1.7765 |
| 7 | ras_to_head_rot_x_90_then_rot_z_90 | 11.621 | 32.340 | 0.2453 | 1.7618 |
| 8 | pca_flip_xy | 12.397 | 35.591 | 0.2274 | 1.7847 |
| 9 | ras_to_head_rot_x_90_then_rot_x_90 | 12.552 | 32.472 | 0.1939 | 1.6903 |
| 10 | ras_to_head | 13.235 | 36.656 | 0.1536 | 1.7811 |
| 11 | ras_to_head_rot_z_90 | 13.661 | 34.561 | 0.2324 | 1.8157 |
| 12 | pca_flip_xz | 13.739 | 13.902 | 0.1458 | 1.7993 |
| 13 | pca_identity | 13.765 | 37.260 | 0.2693 | 1.7231 |
| 14 | ras_to_head_rot_x_90_then_rot_y_-90 | 15.600 | 24.425 | 0.2134 | 1.6478 |
| 15 | ras_to_head_rot_z_-90 | 15.902 | 38.006 | 0.2223 | 1.7496 |
| 16 | ras_to_head_rot_y_90 | 17.004 | 34.258 | 0.0654 | 1.7563 |
| 17 | ras_to_head_rot_y_-90 | 18.697 | 31.503 | 0.0659 | 1.7125 |
| 18 | ras_to_head_rot_y_90_then_rot_y_90 | 23.337 | 34.135 | 0.0000 | 0.0000 |
| 19 | ras_to_head_rot_x_-90_then_rot_z_90 | 23.942 | 26.560 | 0.0609 | 1.7363 |
| 20 | pca_flip_yz | 27.175 | 24.671 | 0.2508 | 1.7517 |
| 21 | ras_to_head_rot_x_90_then_rot_z_-90 | 32.355 | 39.147 | 0.0447 | 1.7772 |
| 22 | ras_to_head_rot_x_-90_then_rot_z_-90 | 34.306 | 47.374 | 0.1547 | 1.7627 |
| 23 | ras_to_head_rot_z_90_then_rot_z_90 | 41.494 | 72.579 | 0.0950 | 1.7928 |

## face_weighted Candidate Ranking

| Rank | Init | Face RMSE | Full RMSE | Fitness | ICP RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | ras_to_head_rot_x_-90 | 3.485 | 6.055 | 0.5175 | 1.7567 |
| 2 | global | 3.506 | 6.783 | 0.4923 | 1.7315 |
| 3 | ras_to_head_rot_x_90 | 6.831 | 8.755 | 0.2463 | 1.7640 |
| 4 | ras_to_head_rot_x_-90_then_rot_y_-90 | 8.633 | 30.620 | 0.1740 | 1.7154 |
| 5 | ras_to_head_rot_y_90_then_rot_z_-90 | 8.982 | 33.253 | 0.2384 | 1.7761 |
| 6 | ras_to_head_rot_x_-90_then_rot_y_90 | 9.295 | 33.051 | 0.1947 | 1.7997 |
| 7 | ras_to_head | 11.978 | 30.062 | 0.1449 | 1.7998 |
| 8 | ras_to_head_rot_x_90_then_rot_y_-90 | 13.640 | 29.675 | 0.1018 | 1.7862 |
| 9 | pca_flip_xz | 14.769 | 15.704 | 0.1513 | 1.8370 |
| 10 | ras_to_head_rot_x_90_then_rot_x_90 | 14.961 | 30.260 | 0.1317 | 1.7782 |
| 11 | pca_flip_yz | 15.193 | 22.568 | 0.1873 | 1.7366 |
| 12 | ras_to_head_rot_z_90 | 16.179 | 34.526 | 0.1515 | 1.7659 |
| 13 | ras_to_head_rot_z_-90 | 19.687 | 33.104 | 0.1480 | 1.7881 |
| 14 | ras_to_head_rot_y_-90 | 22.338 | 29.946 | 0.0720 | 1.8377 |
| 15 | ras_to_head_rot_z_90_then_rot_z_90 | 29.976 | 60.892 | 0.1019 | 1.6792 |
| 16 | ras_to_head_rot_x_-90_then_rot_z_90 | 30.216 | 31.193 | 0.0807 | 1.8450 |
| 17 | ras_to_head_rot_x_90_then_rot_z_-90 | 32.468 | 39.802 | 0.1056 | 1.7554 |
| 18 | ras_to_head_rot_x_90_then_rot_z_90 | 33.577 | 42.241 | 0.2186 | 1.7272 |
| 19 | ras_to_head_rot_x_-90_then_rot_z_-90 | 34.241 | 47.525 | 0.1321 | 1.7797 |
| 20 | ras_to_head_rot_y_90 | 35.798 | 54.454 | 0.1217 | 1.8255 |
| 21 | ras_to_head_rot_y_90_then_rot_y_90 | 36.517 | 61.046 | 0.1169 | 1.7810 |
| 22 | pca_flip_xy | 47.392 | 32.624 | 0.0880 | 1.7848 |
| 23 | pca_identity | 48.594 | 34.995 | 0.0805 | 1.8507 |
