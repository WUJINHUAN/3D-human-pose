# Part of this code is derived/taken from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
import os
import sys
import pickle
import random

import cv2
import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image

from .HumanPoseEstimation import HumanPoseEstimationDataset as Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.transform import fliplr_joints, affine_transform, get_affine_transform
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
from collections import OrderedDict
import logging
from utils.post_processing.nms import oks_nms
from utils.post_processing.nms import soft_oks_nms

logger = logging.getLogger(__name__)
import numpy as np


class COCODataset1(Dataset):
    """
    COCODataset class.
    """

    def __init__(self, root_path="./datasets/COCO", data_version="train2017",
                 is_train=True, use_gt_bboxes=True, bbox_path="",
                 image_width=288, image_height=384,
                 scale=True, scale_factor=0.35, flip_prob=0.5, rotate_prob=0.5, rotation_factor=45., half_body_prob=0.3,
                 use_different_joints_weight=False, heatmap_sigma=3, soft_nms=False):
        """
        Initializes a new COCODataset object.

        Image and annotation indexes are loaded and stored in memory.
        Annotations are preprocessed to have a simple list of annotations to iterate over.

        Bounding boxes can be loaded from the ground truth or from a pickle file (in this case, no annotations are
        provided).

        Args:
            root_path (str): dataset root path.
                Default: "./datasets/COCO"
            data_version (str): desired version/folder of COCO. Possible options are "train2017", "val2017".
                Default: "train2017"
            is_train (bool): train or eval mode. If true, train mode is used.
                Default: True
            use_gt_bboxes (bool): use ground truth bounding boxes. If False, bbox_path is required.
                Default: True
            bbox_path (str): bounding boxes pickle file path.
                Default: ""
            image_width (int): image width.
                Default: 288
            image_height (int): image height.
                Default: ``384``
            color_rgb (bool): rgb or bgr color mode. If True, rgb color mode is used.
                Default: True
            scale (bool): scale mode.
                Default: True
            scale_factor (float): scale factor.
                Default: 0.35
            flip_prob (float): flip probability.
                Default: 0.5
            rotate_prob (float): rotate probability.
                Default: 0.5
            rotation_factor (float): rotation factor.
                Default: 45.
            half_body_prob (float): half body probability.
                Default: 0.3
            use_different_joints_weight (bool): use different joints weights.
                If true, the following joints weights will be used:
                [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5]
                Default: False
            heatmap_sigma (float): sigma of the gaussian used to create the heatmap.
                Default: 3
            soft_nms (bool): enable soft non-maximum suppression.
                Default: False
        """
        super(COCODataset1, self).__init__()

        self.root_path = root_path
        self.data_version = data_version
        self.is_train = is_train
        self.use_gt_bboxes = use_gt_bboxes
        self.bbox_path = bbox_path
        self.scale = scale  # ToDo Check
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight  # ToDo Check
        self.heatmap_sigma = heatmap_sigma
        self.soft_nms = soft_nms

        # Image & annotation path
        self.data_path = f"{root_path}/{data_version}"
        self.annotation_path = f"{root_path}/annotations/person_keypoints_{data_version}.json"
        self.image_set = 'train2017'

        self.image_size = (image_width, image_height)
        self.aspect_ratio = image_width * 1.0 / image_height

        self.heatmap_size = (int(image_width / 4), int(image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200  # I don't understand the meaning of pixel_std (=200) in the original implementation

        self.num_joints = 17
        self.num_joints_half_body = 8

        # eye, ear, shoulder, elbow, wrist, hip, knee, ankle
        self.flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
        self.upper_body_ids = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.lower_body_ids = [0, 1, 2, 3, 4, 5, 6]
        self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5,
                                       1., 1., 1.2, 1.2, 1.5, 1.5]).reshape((self.num_joints, 1))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load COCO dataset - Create COCO object then load images and annotations
        self.coco = COCO(self.annotation_path)

        # Create a list of annotations and the corresponding image (each image can contain more than one detection)

        """ Load bboxes and joints
        - if self.use_gt_bboxes -> Load GT bboxes and joints
        - else -> Load pre-predicted bboxes by a detector (as YOLOv3) and null joints 
        """

        if not self.use_gt_bboxes:
            """
            bboxes must be saved as the original COCO annotations
            i.e. the format must be:
             bboxes = {
               '<imgId>': [
                 {
                   'id': <annId>,  # progressive id for debugging
                   'clean_bbox': np.array([<x>, <y>, <w>, <h>])}
                 },
                 ...
               ],
               ...
             }
            """
            with open(self.bbox_path, 'rb') as fd:
                bboxes = pickle.load(fd)

        self.data = []
        # load annotations for each image of COCO
        indx = self.coco.getImgIds()
        for imgId in tqdm(self.coco.getImgIds(), desc="Prepare images, annotations ... "):

            ann_ids = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)  # annotation ids
            img = self.coco.loadImgs(imgId)[0]  # load img

            if self.use_gt_bboxes:
                objs = self.coco.loadAnns(ann_ids)

                # sanitize bboxes
                valid_objs = []
                for obj in objs:
                    # Skip non-person objects (it should never happen)
                    if obj['category_id'] != 1:
                        continue

                    # ignore objs without keypoints annotation
                    if max(obj['keypoints']) == 0:
                        continue

                    x, y, w, h = obj['bbox']
                    x1 = np.max((0, x))
                    y1 = np.max((0, y))
                    x2 = np.min((img['width'] - 1, x1 + np.max((0, w - 1))))
                    y2 = np.min((img['height'] - 1, y1 + np.max((0, h - 1))))

                    # Use only valid bounding boxes
                    if x2 >= x1 and y2 >= y1:
                        obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                        valid_objs.append(obj)

                objs = valid_objs

            else:
                objs = bboxes[imgId]

            # for each annotation of this image, add the formatted annotation to self.data
            for obj in objs:
                joints = np.zeros((self.num_joints, 2), dtype=np.float)
                joints_visibility = np.ones((self.num_joints, 2), dtype=np.float)

                if self.use_gt_bboxes:
                    """ COCO pre-processing

                    - Moved above
                    - Skip non-person objects (it should never happen)
                    if obj['category_id'] != 1:
                        continue

                    # ignore objs without keypoints annotation
                    if max(obj['keypoints']) == 0:
                        continue
                    """

                    for pt in range(self.num_joints):
                        joints[pt, 0] = obj['keypoints'][pt * 3 + 0]
                        joints[pt, 1] = obj['keypoints'][pt * 3 + 1]
                        t_vis = int(np.clip(obj['keypoints'][pt * 3 + 2], 0, 1))  # ToDo check correctness
                        """
                        - COCO:
                          if visibility == 0 -> keypoint is not in the image.
                          if visibility == 1 -> keypoint is in the image BUT not visible (e.g. behind an object).
                          if visibility == 2 -> keypoint looks clearly (i.e. it is not hidden).
                        """
                        joints_visibility[pt, 0] = t_vis
                        joints_visibility[pt, 1] = t_vis

                center, scale = self._box2cs(obj['clean_bbox'][:4])
                file_name = obj['file_name']
                self.data.append({
                    'imgId': imgId,
                    'annId': obj['id'],
                    'imgPath': f"{self.root_path}/{self.data_version}/{file_name}.jpg",
                    # 'imgPath': os.path.join(self.root_path, self.data_version, 'id_%s.jpg' % imgId),
                    'center': center,
                    'scale': scale,
                    'joints': joints,
                    'joints_visibility': joints_visibility,
                    'bbox': obj['clean_bbox'],
                    'img_size': (img['width'], img['height']),
                    'file_name': f"{file_name}.jpg"
                })

        # Done check if we need prepare_data -> We should not
        print('\nCOCO dataset loaded!')

        # Default values
        self.bbox_thre = 1.0
        self.image_thre = 0.0
        self.in_vis_thre = 0.2
        self.nms_thre = 1.0
        self.oks_thre = 0.9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        joints_data = self.data[index].copy()

        # Load image
        try:
            image = np.array(Image.open(joints_data['imgPath']))
        except:
            raise ValueError(f"Fail to read {joints_data['imgPath']}")

        joints = joints_data['joints']
        joints_vis = joints_data['joints_visibility']

        c = joints_data['center']
        s = joints_data['scale']
        score = joints_data['score'] if 'score' in joints_data else 1
        r = 0

        # Apply data augmentation
        if self.is_train:
            if self.half_body_prob and random.random() < self.half_body_prob and np.sum(
                    joints_vis[:, 0]) > self.num_joints_half_body:
                c_half_body, s_half_body = self._half_body_transform(joints, joints_vis)

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor

            if self.scale:
                s = s * np.clip(random.random() * sf + 1, 1 - sf, 1 + sf)  # A random scale factor in [1 - sf, 1 + sf]

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)  # A random rotation factor in [-2 * rf, 2 * rf]
            else:
                r = 0

            '''if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1'''

        # Apply affine transform on joints and image
        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image_transformed = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )
        # image_transformed = cv2.resize(image, self.image_size)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        '''scale_x = self.image_size[0] / image.shape[1]
        scale_y = self.image_size[1] / image.shape[0]

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.:
                joints[i, 0] = joints[i, 0]*scale_x
                joints[i, 1] = joints[i, 1]*scale_y'''

        self.testimage = False
        if self.testimage == True:
            colors = [(204, 204, 0), (51, 153, 51), (51, 153, 255)]  # 标注关节点的颜色

            def showAnns(img, keypoints):
                for j in range(keypoints.shape[0]):
                    x = keypoints[j][0]
                    y = keypoints[j][1]
                    cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)  # 画点

                cv2.imshow('keypoints', img)
                cv2.waitKey(2000)

            kpts = []
            # img = cv2.imread(image)
            kpts = joints
            showAnns(image_transformed, kpts)

        # Convert image to tensor and normalize
        '''if self.transform is not None:'''  # I could remove this check
        # if image_transformed.shape == (self.image_size[0],self.image_size[1], 3):
        image_transformed = self.transform(image_transformed)
        '''else:
                image_transformed = cv2.cvtColor(image_transformed, cv2.COLOR_GRAY2BGR)
                image_transformed = self.transform(image_transformed)'''

        target, target_weight = self._generate_target(joints, joints_vis)

        # Update metadata
        joints_data['joints'] = joints
        joints_data['joints_visibility'] = joints_vis
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['score'] = score

        return image_transformed, joints.astype(np.float32), target_weight.astype(np.float32), joints_data

    # Private methods
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if random.random() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def _generate_target(self, joints, joints_vis):
        """
        :param joints:  [num_joints, 2]
        :param joints_vis: [num_joints, 2]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': 1,  # 1 == 'person'
                'cls': 'person',
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints'] for k in range(len(img_kpts))], dtype=np.float32)
            key_points = np.zeros((_key_points.shape[0], self.num_joints * 3), dtype=np.float32)

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'].astype(np.float32),
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale']),
                    # 'filename': img_kpts[k]['filename']
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):

        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')

        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

    def evaluate(self, preds, output_dir, all_boxes, img_path, name, *args, **kwargs):
        rank = 0  # 0

        res_folder = os.path.join(output_dir, 'results')  # 文件夹
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )
        # 检测到的文件

        # person x (keypoints)
        _kpts = []
        print('123',int(img_path[0]))
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx]),
                # 'filename': name[idx]
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)  # 'image'是图片id

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = 0.2  # self.in_vis_thre = cfg.TEST.IN_VIS_THRE=0.2
        oks_thre = 0.9  # self.oks_thre = cfg.TEST.OKS_THRE=0.9
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]  # 每个图片的注释
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:  # SOFT_NMS = False
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set:  # val2017
            print('yongle')
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0


if __name__ == '__main__':
    # from skimage import io
    coco = COCODataset1(root_path=f"{os.path.dirname(__file__)}/COCO", data_version="train2017", rotate_prob=0.,
                        half_body_prob=0.)
    item = coco[1]
    # io.imsave("tmp.jpg", item[0].permute(1,2,0).numpy())
    print()
    print(item[1].shape)
    print('ok!!')
    # img = np.clip(np.transpose(item[0].numpy(), (1, 2, 0))[:, :, ::-1] * np.asarray([0.229, 0.224, 0.225]) +
    #     np.asarray([0.485, 0.456, 0.406]), 0, 1) * 255
    # cv2.imwrite('./tmp.png', img.astype(np.uint8))
    # print(item[-1])
    pass
