import json
from os.path import dirname, join, realpath

config = json.load(open(join(dirname(realpath(__file__)), 'config.json')))
ratio = config['ratio']
segment_directory = config['segment_directory']
image_directory = config['image_directory']