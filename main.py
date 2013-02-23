#!/usr/bin/env python
# coding: utf-8

import scipy as sp
from jet import JetDescriptor
from util import read_keypoints, write_keypoints
import argparse

description = 'Extract jet descriptors from an image using a set of keypoints.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--image_file', type=str, required=True,
                    help='Image file path.')
parser.add_argument('-k', '--keypoint_file', type=str, required=True,
                    help='Keypoints file path.')
parser.add_argument('-o', '--output_file', type=str, required=True,
                    help='Output file path')
args = parser.parse_args()


def run():
    img = sp.misc.imread(args.image_file, flatten=True)/255.
    keypoints = read_keypoints(args.keypoint_file)
    jd = JetDescriptor()
    descs = jd.compute(img, keypoints)
    write_keypoints(args.output_file, keypoints, descs)

if __name__ == '__main__':
    run()
