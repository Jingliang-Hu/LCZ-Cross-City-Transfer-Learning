# Experiments
## Parameter setting
In each script, parameters are saved in "paraDict".
## Re-run
Example with upper boundary.
```bash
cd 0_upper_boundary
python baseline_upper_bound.py
```
## Output
Each run will create a folder whose name has main parameters and a time stamp. Values of "paraDict" would also be saved in the folder.

# Description
This folder contains all implemented algorithms in the paper.
* 0_upper_boundary
  - A practical demonstration of the upper boundary for transfer learning methods. A baseline model trained and tested with samples from the target domain
* 1_baseline
  - A baseline model trained with samples from the source domain and tested with samples from the target domain
* 2_self_ensemble
  - "French, G., Mackiewicz, M., Fisher, M., 2017. Self-ensembling for visual domain adaptation. arXiv preprint arXiv:1706.05208"
* 3_dual_student
  - "Ke, Z., Wang, D., Yan, Q., Ren, J., Lau, R.W., 2019. Dual student: Breaking the limits of the teacher in semi-supervised learning, in: Proceedings of the IEEE International Conference on Computer Vision, pp. 6728-6736."
* 3_dual_student_aug
  - "Ke, Z., Wang, D., Yan, Q., Ren, J., Lau, R.W., 2019. Dual student: Breaking the limits of the teacher in semi-supervised learning, in: Proceedings of the IEEE International Conference on Computer Vision, pp. 6728-6736." Data Augmentation Version
* 4_our_model
  - Our model
* 9_amran
  - "Zhu, S., Du, B., Zhang, L., Li, X., 2021. Attention-based multiscale residual adaptation network for cross-scene classifcation. IEEE Transactions on Geoscience and Remote Sensing, doi:10.1109/TGRS.2021.3056624."
  - Implementation from "https://github.com/WHUzhusihan96/AMRAN"
* 9_pesudo_label
  - "Tong, X.Y., Xia, G.S., Lu, Q., Shen, H., Li, S., You, S., Zhang, L., 2020. Land-cover classification with high-resolution remote sensing images using transferable deep models. Remote Sensing of Environment 237, 111322."
* 101_baseline_data_fusion
* 101_baseline_decision_fusion
* 101_baseline_feature_fusion
* 101_our_fusion

