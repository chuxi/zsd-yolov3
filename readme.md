using `tensorflow 2.0.0-beta1`


```shell script
# download yolov3.weights

wget https://pjreddie.com/media/files/yolov3.weights
```

```shell script
python convert.py yolov3.cfg yolov3.weights data/yolo3_weights.h5
```

downloaded files:

```shell script
ZJLAB_ZSD_2019/
    train/
        img_0000xxxx.jpg
    instances_train.json
    seen_embeddings_GloVe.json
ZJLAB_ZSD_2019_pretest/
    img_0000xxxx.jpg
clsname2id_20k.json
unseen_GloVe.json
```

```shell script
python zsd_train.py
```

```shell script
python zsd_test.py
```
