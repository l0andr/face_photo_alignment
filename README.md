# face_photo_alignment
<h3>Python Implementation of CP2Tform and its Application for Aligning Face Photos Using Five Points.</h3>

### Instalation:

pip install -r requirements.txt --user

### Run unit-tests:  

python -m pytest tests

### Usage as command line tool:
#### python face_align_512.py [-h] -i I -p P -o O [--show]

Arguments:
  -h, --help  show help message and exit <br>
  -i I        Path to image file <br>
  -p P        Path to text files with face 5 points <br>
  -o O        Path to output directory <br>
  --show      Show images [optional]


### Usage as python module

```
from face_align_512 import face_align

face_align(facial5point,img,coord5point,imgSize)
    """
    :param facial5point: 5x2 np.ndarray wth coordinates of 5 face points
    :param img: np.ndarray with image for transformation
    :param coord5point: [optional] coresponded five points on transformed image
    :param imgSize: [optional] output image size - default 512
    :return:
    (trans_img,trans_points)
    trans_img - image after transformation
    trans_points - np.ndarray 5x2 with facial5point after transformation
    """
```
