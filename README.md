# STANet:Spatiotemporal Adaptive Feature Modeling for Accurate Human Motion Prediction
## Abstract
Human motion prediction is crucial for enabling machines and intelligent agents to effectively interact with their surroundings. Despite significant progress, existing methods based on Graph Convolutional Networks (GCNs) still struggle to efficiently capture and integrate temporal and spatial features. In this paper, we propose STANet, a novel Spatiotemporal Adaptive Network that dynamically captures inter-body joint correlations through temporal features to enhance spatial information modeling in GCNs. STANet introduces a Spatial Correlation Reasoning (SCR) module to discover temporal correlations and coordination features, utilizing predefined adjacency matrices to flexibly capture temporal correlations of limb movements. An innovative spatiotemporal feature cross-extraction strategy is applied to improve the efficiency of feature fusion, reducing the risk of local optima and enhancing prediction accuracy. Extensive experiments on two challenging datasets demonstrate that STANet outperforms existing methods, validating its effectiveness and superiority in spatiotemporal feature modeling for human motion prediction.

## How to use
```
python main_3d.py --data_dir [Path To Your H36M data] --input_n 10 --output_n 10 --dct_n 20 --exp [Path To Your H36M model]
```
```
python main_3d.py --data_dir [Path To Your H36M data] --input_n 10 --output_n 25 --dct_n 35 --exp [Path To Your H36M model]
```
```
python main_cmu_3d.py --data_dir_cmu [Path To Your CMU data] --input_n 10 --output_n 10 --dct_n 20 --exp [Path To Your CMU model]
```
