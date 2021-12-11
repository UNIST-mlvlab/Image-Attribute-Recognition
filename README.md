# Image Attribute Recognition
Image attribute recognition code implementation based on ALM model [[1](https://arxiv.org/pdf/1910.04562.pdf)]

### Requirements
```shell
pip install -r requirements.txt
```

### Usages
1. Download the trained model from [here](https://unistackr0-my.sharepoint.com/:u:/g/personal/suyeong_unist_ac_kr/EaIqVdgEDa5MmN1wP07kwuEB1jtwZyYWiNzu3KgGa9btIg?e=9ZxHCK).
2. Put the model `saved_model.pt` file under `./model` or `./app/alm/model`
3. Train
	```shell
	python main.py --experiment=rap
	```
4. Visualize
	```shell
	python main.py --experiment=rap --vis
	```
    Start TensorBoard with ```tensorboard --logdir ./runs```

### Reference
[1] *Tang, C., Sheng, L., Zhang, Z., & Hu, X. (2019). Improving pedestrian attribute recognition with weakly-supervised multi-scale attribute-specific localization. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ (pp. 4997-5006).*
