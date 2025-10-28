import yaml
import os
import pprint

bag_path = os.path.expanduser("~/ros2_ws/bags/CAST/collect5/bag_20241213_162052_collect5/metadata.yaml")

with open(bag_path, 'r') as f:
    metadata = yaml.safe_load(f)

print("\n🧾 rosbag2_bagfile_information contents:\n")
pprint.pprint(metadata['rosbag2_bagfile_information'])
