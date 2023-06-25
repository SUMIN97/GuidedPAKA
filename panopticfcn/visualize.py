import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import torch.nn.functional as F
import cv2

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


class UseDetectronVisualize:
    def __init__(self):
        self.stuff_meta = MetadataCatalog.get('coco_2017_train_panoptic_separated')
        self.panoptic_meta = MetadataCatalog.get('coco_2017_train_panoptic')
        self.instance_mode = ColorMode(2)

    def show_stuff(self, image, pd_reg, file_name, save_dir):
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        image = image.astype(np.uint8)

        pd_reg = pd_reg.detach().cpu().numpy()
        vis = Visualizer(image, self.stuff_meta, scale=1.2)
        vis.draw_sem_seg(pd_reg).save(os.path.join(save_dir, file_name + "_pd_reg.png"))

    def show_semantic(self, image, gt_sem, pd_sem, file_name, save_dir):
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        image = image.astype(np.uint8)

        pd_sem = pd_sem.detach().cpu().numpy()
        vis = Visualizer(image, self.panoptic_meta, scale=1.2)
        vis.draw_sem_seg(pd_sem).save(os.path.join(save_dir, file_name + "_pd_sem.png"))

        gt_sem = gt_sem.detach().cpu().numpy()
        vis = Visualizer(image, self.panoptic_meta, scale=1.2)
        vis.draw_sem_seg(gt_sem).save(os.path.join(save_dir, file_name + "_gt_sem.png"))





class Visualize:
    def __init__(self, train_dataset):

        self.meta = MetadataCatalog.get(train_dataset)
        self.name = self.meta.get('name')
        if train_dataset == 'coco_2017_train_panoptic':
            self.category_id_colors = np.array(self.meta.get('stuff_colors'))
        elif train_dataset == 'coco_2017_train_panoptic_separated':
            self.category_id_colors = np.array(self.meta.get('thing_colors') + self.meta.get('stuff_colors')[1:])

        self.n_classes = len(self.category_id_colors)
        self.plot = False

    def show_target(self, target, pan_seg_file_name):
        file_name = os.path.basename(pan_seg_file_name).split('.')[0]
        semantic = target['semantic'].detach().cpu().numpy()
        semantic[semantic == 255] = 0
        self.decode_segmap(semantic, save_path='tmp/{}_semantic.png'.format(file_name))

        inst_bitmasks = target['inst_bitmasks'].detach().cpu().numpy()
        inst_classes = target['inst_classes'].detach().cpu().numpy()
        stuff_bitmasks = target['stuff_bitmasks'].detach().cpu().numpy()
        stuff_classes = target['stuff_classes'].detach().cpu().numpy()


        for i in range(100):
            b = inst_bitmasks[i]
            c = inst_classes[i]
            if c == 255: break
            Image.fromarray((b * 255).astype(np.uint8)).save('tmp/{}_{}_{}.png'.format(file_name, self.total_classes[int(c)], i))
        for i in range(100):
            b = stuff_bitmasks[i]
            c = stuff_classes[i]
            if c == 255: break
            Image.fromarray((b * 255).astype(np.uint8)).save('tmp/{}_{}_{}.png'.format(file_name, self.total_classes[int(c)], i))

        return

    def show_image(self, file_name, image, save_dir):
        image = image.permute(1, 2, 0)
        image = image.detach().cpu().numpy()
        image = image.astype(np.uint8)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_image).save(os.path.join(save_dir, file_name+"_image.png"))

    def show_neighbor(self, file_name, gt_neighbor, pd_neighbor, save_dir):
        mask = gt_neighbor >= 0
        gt_neighbor = gt_neighbor[mask]
        pd_neighbor = pd_neighbor[mask]

        gt_neighbor = gt_neighbor.detach().cpu().numpy()
        # pd_neighbor = F.interpolate(pd_neighbor.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]
        pd_neighbor = pd_neighbor.detach().cpu().numpy()
        gt = gt_neighbor.flatten()
        pred = pd_neighbor.flatten()
        assert len(gt) == len(pred)

        #calculate roc curves
        fpr,tpr, threshold = metrics.roc_curve(gt, pred)
        #get the best threshold
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = threshold[ix]
        sens, spec = tpr[ix], 1 - fpr[ix]

        roc_auc = metrics.auc(fpr, tpr)
        plt.figure()
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.scatter(fpr[ix], tpr[ix], marker='+', s=100, color='r', label='Best threshold = %.3f, \nSensitivity = %.3f \nSpecifity = %.3f' % (best_thresh, sens, spec))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        if self.plot == True:
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, file_name + '_roc.png'))
        plt.close()

    def show_semantic(self, file_name, gt_semantic, pred_semantic, save_dir):
        gt_semantic = gt_semantic.detach().cpu().numpy()
        # pred_semantic = F.interpolate(pred_semantic.unsqueeze(0), scale_factor=4, mode="bilinear", align_corners=False)[0]
        pred_semantic = pred_semantic.detach().cpu().numpy()

        save_path = os.path.join(save_dir, file_name + '_gt.png')
        self.decode_segmap(gt_semantic.argmax(axis=0), save_path)
        save_path = os.path.join(save_dir, file_name + '_pd.png')
        self.decode_segmap(pred_semantic.argmax(axis=0), save_path)


    def show_center(self,  file_name, gt_center, pd_center, save_dir):
        pd_center = pd_center.sigmoid()
        gt_center = gt_center.detach().cpu().numpy()
        pd_center = pd_center.detach().cpu().numpy()
        gt_center = np.clip(gt_center, a_min=0.0, a_max=1.0)
        pd_center = np.clip(pd_center, a_min=0.0, a_max=1.0)

        gt_center *= 255.0
        pd_center *= 255.0

        for i in range(self.num_thing_classes):
            Image.fromarray(gt_center[i].astype(np.uint8)).save(os.path.join(save_dir, file_name + '_cen_{}_gt.png'.format(i+1)))
            Image.fromarray(pd_center[i].astype(np.uint8)).save(os.path.join(save_dir, file_name + '_cen_{}_pd.png'.format(i+1)))

    def decode_segmap(self, label_mask, save_path=None):

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = self.category_id_colors[ll, 0]
            g[label_mask == ll] = self.category_id_colors[ll, 1]
            b[label_mask == ll] = self.category_id_colors[ll, 2]

        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))

        if self.plot:
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            plt.figure()
            plt.imshow(rgb)
            plt.show()
        else:
            rgb[:, :, 0] = r
            rgb[:, :, 1] = g
            rgb[:, :, 2] = b

            Image.fromarray(rgb.astype(np.uint8)).save(save_path)


if __name__ == '__main__':
    a = Visualize(train_dataset='coco_2017_train_panoptic', save_dir='tmp')
