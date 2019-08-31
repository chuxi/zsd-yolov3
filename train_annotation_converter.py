import json


if __name__ == '__main__':
    train_annotation_file = "data/train.txt"
    train_instance_file = "ZJLAB_ZSD_2019/instances_train.json"
    with open(train_instance_file, mode='r') as f:
        instances_json = json.load(f)
    image_bbox = {}
    for item in instances_json["annotations"]:
        image_id = item["image_id"]
        if image_id not in image_bbox:
            # write file name
            image_file_name = "img_%08d.jpg" % image_id
            image_bbox[image_id] = []
            image_bbox[image_id].append("ZJLAB_ZSD_2019/train/" + image_file_name)
        bbox = item["bbox"]
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox.append(item["category_id"])
        image_bbox[image_id].append(bbox)

    with open(train_annotation_file, "w") as f:
        for value in image_bbox.values():
            f.write(value[0])
            for a in value[1:]:
                f.write(" " + ",".join(str(int(b)) for b in a))
            f.write("\n")
    print("total images: ", len(image_bbox))
