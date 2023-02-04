import argparse
import cp2tform
import numpy as np
from PIL import Image
import cv2 as cv
import os
import matplotlib.pyplot as plt
import PyQt5
from typing import List, Optional

def face_align(facial5point:np.ndarray,img:np.ndarray,coord5point:np.ndarray = None,imgSize:Optional[List[int]]=None):

    if imgSize is None:
        imgSize = [512,512]
    if coord5point is None:
        coord5point = np.array([[180, 230], [300, 230], [240, 301], [186, 365.6], [294, 365.6]])
        coord5point[:,0] = coord5point[:,0] * imgSize[0] / 512
        coord5point[:, 1] = coord5point[:, 1] * imgSize[1] / 512
    if len(imgSize) != 2:
        raise RuntimeError("face_align: imgSize should contain size of output image in list of two elements")
    if facial5point.shape[0] != 5 or facial5point.shape[1] != 2:
        raise RuntimeError("face_align: facial5point should contain np.ndarray with shape 5x2")
    if coord5point.shape[0] != 5 or coord5point.shape[1] != 2:
        raise RuntimeError("face_align: coord5point should contain np.ndarray with shape 5x2")

    transf = cp2tform.cp2tform(facial5point, coord5point, 'similarity')
    trans_img = cv.warpAffine(img, np.transpose(transf[0]['tdata']['T'][:, 0:2]), (imgSize[0], imgSize[1]))
    trans_points = cp2tform.tformfwd_x(transf[0],facial5point)
    return (trans_img,trans_points)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face Photo Align test application",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", help="Path to image file ", type=str, required=True)
    parser.add_argument("-p", help="Path to text files with face points", type=str, required=True)
    parser.add_argument("-o", help="Path to output directory", type=str, required=True)
    parser.add_argument("--show", help="Show images",
                        default=False, action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.i):
        raise RuntimeError(f"Input file '{args.i}' do not exists")

    if not os.path.exists(args.p):
        raise RuntimeError(f"Input file with coordinates of facial points '{args.p}' do not exists")


    if not os.path.isdir(args.o):
        raise RuntimeError(f"Output path  '{args.o}' do not exists")


    impath = args.i
    pointspath = args.p
    facial5point = np.loadtxt(pointspath,delimiter = ',')
    if facial5point.shape[0] != 5 or facial5point.shape[1] != 2:
        raise RuntimeError(f"Seems file {args.p} with facial points have  incorrect format")
    imgSize = [512, 512]
    coord5point = np.array([[180, 230],[300, 230],[240, 301],[186, 365.6],[294, 365.6]])
    coord5point = (coord5point - 240) / 560 * 512 + 256
    Pimage = Image.open(impath)
    img = np.asarray(Pimage)
    trans_img,trans_points=face_align(facial5point, img,coord5point, imgSize)
    trans_points = np.round(trans_points)

    if args.show:
        fig,ax=plt.subplots(ncols=2)
        ax[0].imshow(img)
        ax[1].imshow(trans_img)
        plt.show()
        print(f"Input facial points: {facial5point}")
        print(f"Same points after transformation: {trans_points}")

    [inpdir,inpfile] = os.path.split(impath)
    inpfilename,exten = inpfile.split('.', 1)
    outimgpath = os.path.join(args.o,inpfilename+"_transform"+"."+exten)
    image = Image.fromarray(trans_img)
    image.save(outimgpath)
    outcoordpath = os.path.join(args.o, inpfilename + "_coords.txt")
    np.savetxt(outcoordpath,trans_points,fmt="%.0f")


