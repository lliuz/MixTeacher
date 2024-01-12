from mmdet.models import FasterRCNN, DETECTORS, build_detector
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class SELayer(nn.Module):
    def __init__(self, in_channel=512, output_channel=2, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, output_channel, bias=False),
            nn.Softmax()
        )

        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, -1, 1, 1)
        return y


@DETECTORS.register_module()
class MSInputFasterRCNN(FasterRCNN):
    def __init__(self, divisor=32, test_on='fr', kd_alpha=0., kd_t=3, do_reduce=True,
                 resize_img_only=False, pad_resize_img=True, msi_train=True, *args, **kwargs):
        super(MSInputFasterRCNN, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'External proposals not implemented'
        self.divisor = divisor
        self.SE = SELayer(reduction=16)
        self.test_on = test_on
        self.count = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.KD_T = kd_t
        self.KD_ALPHA = kd_alpha
        self.do_reduce = do_reduce
        self.resize_img_only = resize_img_only
        self.pad_resize_img = pad_resize_img
        self.msi_train = msi_train

    def impad_to_multiple(self, img, pad_val=0):
        pad_h = int(np.ceil(img.shape[2] / self.divisor)) * self.divisor
        pad_w = int(np.ceil(img.shape[3] / self.divisor)) * self.divisor
        height = max(pad_h - img.shape[2], 0)
        width = max(pad_w - img.shape[3], 0)
        padding = (0, width, 0, height)
        return F.pad(img, padding, value=pad_val)

    def resize_info(self, hr_info, scale=0.5):
        assert 'img' in hr_info
        ret_info = deepcopy(hr_info)

        ret_info['img'] = F.interpolate(ret_info['img'], scale_factor=scale, mode="nearest")
        if self.pad_resize_img:
            ret_info['img'] = self.impad_to_multiple(ret_info['img'])
        if self.resize_img_only:  # skip align the img_metas and gts
            return ret_info

        if 'img_metas' in hr_info and hr_info['img_metas'] is not None:
            img_metas = ret_info['img_metas']
            hr_metas = hr_info['img_metas']
            for i in range(len(img_metas)):
                h, w, c = hr_metas[i]['img_shape']
                img_metas[i]['img_shape'] = (int(np.round(h * scale + 1e-7)), int(np.round(w * scale + 1e-7)), c)
                h, w, c = hr_metas[i]['pad_shape']
                img_metas[i]['pad_shape'] = (int(np.round(h * scale + 1e-7)), int(np.round(w * scale + 1e-7)), c)
                h, w, _ = img_metas[i]['img_shape']
                h0, w0, _ = img_metas[i]['ori_shape']
                img_metas[i]['scale_factor'] = np.array([w / w0, h / h0, w / w0, h / h0])
                img_metas[i]['batch_input_shape'] = ret_info['img'].shape[2:]
                if 'transform_matrix' in hr_info['img_metas'][i]:
                    scale_mat = np.eye(3)
                    h, w, _ = img_metas[i]['img_shape']
                    h0, w0, _ = hr_metas[i]['img_shape']
                    scale_mat[0, 0] = w / w0
                    scale_mat[1, 1] = h / h0
                    img_metas[i]['transform_matrix'] = scale_mat @ hr_info['img_metas'][i]['transform_matrix']

        for key in ['gt_bboxes', 'gt_bboxes_ignore']:
            if hr_info.get(key, None):
                for i in range(len(ret_info[key])):
                    scale_factor = torch.from_numpy(img_metas[i]['scale_factor'] /
                                                    hr_info['img_metas'][i]['scale_factor'])
                    ret_info[key][i] *= scale_factor.type_as(ret_info[key][i])
        return ret_info

    def extract_fuse_feat(self, hr_features, lr_features):
        fr_features = [hr_features[0]]
        for x_hr, x_lr in zip(hr_features[1:], lr_features[:-1]):  # P3~P6 x P2~P5
            h, w = x_hr.shape[-2:]
            x_lr = x_lr[:, :, :h, :w]
            score = self.SE(torch.cat([x_hr, x_lr], dim=1))
            x_fr = score[:, 0].unsqueeze(-1) * x_hr + score[:, 1].unsqueeze(-1) * x_lr
            fr_features.append(x_fr)
        return fr_features

    def kd(self, t_features, s_features):
        d_fpn_loss = 0
        for index, (t_feature, s_feature) in enumerate(zip(t_features, s_features)):
            tt_feature = (t_feature / self.KD_T)
            ss_feature = (s_feature / self.KD_T)
            d_fpn_loss += F.l1_loss(
                ss_feature, tt_feature).mean() * self.KD_ALPHA * self.KD_T * self.KD_T * min(1.0, self.count / 500.0)
        return d_fpn_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        hr_info = {'img': img, 'img_metas': img_metas,
                   'gt_bboxes': gt_bboxes, 'gt_bboxes_ignore': gt_bboxes_ignore,
                   'gt_labels': gt_labels, 'gt_masks': gt_masks}
        hr_features = self.extract_feat(hr_info['img'])
        multi_res_info = {
            'hr': dict(x=hr_features, **hr_info),
        }
        if self.msi_train:
            lr_info = self.resize_info(hr_info, scale=0.5)
            lr_features = self.extract_feat(lr_info['img'])
            fr_features = self.extract_fuse_feat(hr_features, lr_features)
            multi_res_info.update({
                'lr': dict(x=lr_features, **lr_info),
                'fr': dict(x=fr_features, **hr_info)
            })
        return self.multi_res_losses(multi_res_info, **kwargs)

    def multi_res_losses(self, multi_res_info, **kwargs):
        losses = dict()

        for res_key, info in multi_res_info.items():
            x, img_metas = info['x'], info['img_metas']
            gt_bboxes, gt_labels = info['gt_bboxes'], info['gt_labels']
            gt_bboxes_ignore, gt_masks = info['gt_bboxes_ignore'], info['gt_masks']
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update({k + f'_{res_key}': v for k, v in rpn_losses.items()})

            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks,
                                                     **kwargs)
            losses.update({k + f'_{res_key}': v for k, v in roi_losses.items()})

        # 全部相加还是都除以基数?
        losses = self.reduce_losses(losses, len(multi_res_info))
        self.count += 1
        return losses

    def reduce_losses(self, losses, n_res):
        if not self.do_reduce:
            return losses
        for loss_name, loss_value in losses.items():
            if 'loss' not in loss_name:
                continue
            if isinstance(loss_value, torch.Tensor):
                losses[loss_name] = loss_value / n_res
            elif isinstance(loss_value, list):
                losses[loss_name] = [_loss / n_res for _loss in loss_value]
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        return losses

    def extract_test_feat(self, img, img_metas=None):
        hr_info = {'img': img, 'img_metas': img_metas}
        if self.test_on == 'hr':
            x = self.extract_feat(hr_info['img'])
            return x, hr_info['img_metas']
        elif self.test_on == 'lr':
            lr_info = self.resize_info(hr_info, scale=0.5)
            x = self.extract_feat(lr_info['img'])
            return x, lr_info['img_metas']
        elif self.test_on == 'fr':
            lr_info = self.resize_info(hr_info, scale=0.5)
            hr_features = self.extract_feat(hr_info['img'])
            lr_features = self.extract_feat(lr_info['img'])
            x = self.extract_fuse_feat(hr_features, lr_features)
            return x, hr_info['img_metas']
        else:
            raise ValueError(f'test_on: {self.test_on}')

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x, img_metas = self.extract_test_feat(img, img_metas)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
