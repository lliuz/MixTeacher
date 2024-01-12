import copy

import torch
import torch.nn as nn
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps, build_assigner, multiclass_nms
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid


@DETECTORS.register_module()
class SoftTeacherSimMSI(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacherSimMSI, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

            initial_assigner_cfg = dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1)
            self.initial_assigner = build_assigner(initial_assigner_cfg)
        self.use_sigmoid = self.student.roi_head.bbox_head.use_sigmoid
        self.num_classes = self.student.roi_head.bbox_head.num_classes
        if not self.use_sigmoid:
            self.num_classes += 1

        self.unsup_start = 5000
        self.unsup_warmup = 2000

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)

        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")
        loss = {}
        # ! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        # ! it means that at least one sample for each group should be provided on each gpu.
        # ! In some situation, we can only put one image per gpu, we have to return the sum of loss
        # ! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes),
                 "cur_iter": self.cur_iter},
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            loss_weight = self.unsup_weight
            if self.cur_iter < self.unsup_start + self.unsup_warmup:
                loss_weight *= 0 if self.cur_iter < self.unsup_start \
                    else (self.cur_iter - self.unsup_start) / self.unsup_warmup
            if loss_weight > 0:
                unsup_loss = weighted_loss(
                    self.foward_unsup_train(
                        data_groups["unsup_teacher"], data_groups["unsup_student"]),
                    weight=loss_weight,
                )
                unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
                loss.update(**unsup_loss)
        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        tea_img = teacher_data["img"]
        stu_img = student_data["img"]
        tea_img_metas = teacher_data["img_metas"]
        stu_img_metas = student_data["img_metas"]

        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in tea_img_metas]
        snames = [meta["filename"] for meta in stu_img_metas]
        tidx = [tnames.index(name) for name in snames]
        tea_img = tea_img[torch.Tensor(tidx).to(tea_img.device).long()]
        tea_img_metas = [tea_img_metas[idx] for idx in tidx]

        with torch.no_grad():
            tea_feat, tea_img_metas = self.teacher.extract_test_feat(tea_img, tea_img_metas)
            det_bboxes, det_labels, tea_proposals, tea_feats, tea_bbox_pred, tea_score_pred = self.extract_teacher_info(
                tea_feat, tea_img_metas,
            )

        pseudo_labels = det_labels

        stu_hr_info = {'img': stu_img, 'img_metas': stu_img_metas}
        stu_lr_info = self.student.resize_info(stu_hr_info, scale=0.5)

        stu_hr_features = self.student.extract_feat(stu_hr_info['img'])
        stu_lr_features = self.student.extract_feat(stu_lr_info['img'])
        stu_fr_features = self.student.extract_fuse_feat(stu_hr_features, stu_lr_features)

        multi_res_info = {
            'hr': dict(x=stu_hr_features, **stu_hr_info),
            'lr': dict(x=stu_lr_features, **stu_lr_info),
            'fr': dict(x=stu_fr_features, **stu_hr_info)
        }
        if self.train_cfg.get('mine_pseudo_threshold', -1) > 0:
            multi_res_info['hr'].update(
                dict(ref_features=stu_hr_features, ref_img_metas=stu_hr_info['img_metas'],
                     tgt_features=stu_fr_features, tgt_img_metas=stu_hr_info['img_metas'])
            )

        ms_loss = {}
        for res_key, info in multi_res_info.items():
            loss = {}
            stu_feats, stu_img, stu_img_metas = info['x'], info['img'], info['img_metas']
            pseudo_bboxes = self.convert_bbox_space(tea_img_metas, stu_img_metas, det_bboxes)
            pseudo_proposals = self.convert_bbox_space(tea_img_metas, stu_img_metas, tea_proposals)
            rpn_loss, stu_proposals = self.rpn_loss(stu_feats, pseudo_bboxes, stu_img_metas)
            loss.update(rpn_loss)

            if 'ref_features' not in info:
                tgt_cls_scores, tgt_decoded_bboxes = None, None
                ref_cls_scores, ref_decoded_bboxes = None, None
            else:
                ref_feats, ref_img_metas = info['ref_features'], info['ref_img_metas']
                tgt_feats, tgt_img_metas = info['tgt_features'], info['tgt_img_metas']
                tgt_proposals = self.convert_bbox_space(stu_img_metas, tgt_img_metas, stu_proposals)
                tgt_cls_scores, _, tgt_decoded_bboxes = self.extract_proposal_prediction(
                    tgt_feats, tgt_img_metas, tgt_proposals)
                tgt_decoded_bboxes = self.convert_bbox_space(tgt_img_metas, stu_img_metas, tgt_decoded_bboxes)

                ref_proposals = self.convert_bbox_space(stu_img_metas, ref_img_metas, stu_proposals)
                ref_cls_scores, _, ref_decoded_bboxes = self.extract_proposal_prediction(
                    ref_feats, ref_img_metas, ref_proposals)
                ref_decoded_bboxes = self.convert_bbox_space(ref_img_metas, stu_img_metas, ref_decoded_bboxes)

            if self.train_cfg.use_teacher_proposal:
                stu_proposals = pseudo_proposals

            loss.update(
                self.unsup_rcnn_cls_loss(
                    stu_feats,
                    stu_img_metas,
                    stu_proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                    tea_img_metas,
                    tea_feats,
                    ref_decoded_bboxes,
                    ref_cls_scores,
                    tgt_decoded_bboxes,
                    tgt_cls_scores
                )
            )
            loss.update(
                self.unsup_rcnn_reg_loss(
                    stu_feats,
                    stu_img_metas,
                    stu_proposals,
                    pseudo_bboxes,
                    pseudo_labels,
                )
            )
            ms_loss.update({k + f'_{res_key}': v for k, v in loss.items()})
        ms_loss = self.student.reduce_losses(ms_loss, len(multi_res_info))
        return ms_loss

    def rpn_loss(self, stu_feats, pseudo_bboxes, img_metas, gt_bboxes_ignore=None, ):
        rpn_out = self.student.rpn_head(stu_feats)
        rpn_out = list(rpn_out)
        gt_bboxes = []
        for bbox in pseudo_bboxes:
            bbox, _, _ = filter_invalid(
                bbox[:, :4],
                score=bbox[
                      :, 4
                      ],  # TODO: replace with foreground score, here is classification score,
                thr=self.train_cfg.rpn_pseudo_threshold,
                min_size=self.train_cfg.min_pseduo_box_size,
            )
            gt_bboxes.append(bbox)
        log_every_n(
            {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
        losses = self.student.rpn_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
        )
        proposal_cfg = self.student.train_cfg.get(
            "rpn_proposal", self.student.test_cfg.rpn
        )
        proposal_list = self.student.rpn_head.get_bboxes(
            *rpn_out, img_metas=img_metas, cfg=proposal_cfg
        )
        return losses, proposal_list

    def unsup_rcnn_cls_loss(
            self,
            feat,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            teacher_img_metas,
            teacher_feat,
            ref_bboxes,
            ref_scores,
            tgt_bboxes,
            tgt_scores
    ):
        sampling_results, gt_bboxes, gt_labels = self.get_sampling_result_mine(
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            ref_bboxes,
            ref_scores,
            tgt_bboxes,
            tgt_scores,
            gt_bboxes_ignore=None,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        # modify the negative proposal target
        aligned_proposals = self.convert_bbox_space(img_metas, teacher_img_metas, selected_bboxes)
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        return loss

    def unsup_rcnn_reg_loss(
            self,
            feat,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels
        )["loss_bbox"]
        return {"loss_bbox": loss_bbox}

    def get_sampling_result_mine(
            self,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            ref_bboxes,
            ref_scores,
            tgt_bboxes,
            tgt_scores,
            gt_bboxes_ignore=None,
    ):
        if ref_bboxes is None:
            return self.get_sampling_result_original(img_metas, proposal_list, gt_bboxes, gt_labels)

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            gt_bboxes_per_img = gt_bboxes[i][:, :4]
            gt_scores_per_img = gt_bboxes[i][:, 4]
            gt_labels_per_img = gt_labels[i]

            bboxes_ind = gt_scores_per_img > self.train_cfg.cls_pseudo_threshold
            gt_bboxes_per_img_high = gt_bboxes_per_img[bboxes_ind]
            gt_labels_per_img_high = gt_labels_per_img[bboxes_ind]

            # general label assignment with high confidence pseudo boxes
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes_per_img_high, gt_bboxes_ignore[i], gt_labels_per_img_high)

            # mine positive for all proposals assigned to negative
            neg_inds = assign_result.gt_inds == 0
            ref_bboxes_neg_per_img = ref_bboxes[i][neg_inds]
            tgt_bboxes_neg_per_img = tgt_bboxes[i][neg_inds]
            ref_scores_neg_per_img = ref_scores[i][neg_inds]
            tgt_scores_neg_per_img = tgt_scores[i][neg_inds]

            bboxes_ind = torch.bitwise_and(
                gt_scores_per_img < self.train_cfg.cls_pseudo_threshold,
                gt_scores_per_img > self.train_cfg.mine_pseudo_threshold
            )
            gt_bboxes_per_img_low = gt_bboxes_per_img[bboxes_ind]
            gt_labels_per_img_low = gt_labels_per_img[bboxes_ind]

            # assign reference bboxes with low confidence pseudo boxes
            assign_result_low = self.initial_assigner.assign(
                ref_bboxes_neg_per_img, gt_bboxes_per_img_low, None, gt_labels_per_img_low)

            # for all ref bboxes assigned to positive
            gt_inds = assign_result_low.gt_inds
            pos_inds = gt_inds > 0

            assigned_gt_inds = gt_inds - 1
            pos_assigned_gt_inds = assigned_gt_inds[pos_inds]
            pos_labels = gt_labels_per_img_low[pos_assigned_gt_inds]
            pos_tgt_scores_per_img = tgt_scores_neg_per_img[pos_inds]
            pos_ref_scores_per_img = ref_scores_neg_per_img[pos_inds]
            pos_tgt_bboxes_per_img = tgt_bboxes_neg_per_img[pos_inds]
            pos_ref_bboxes_per_img = ref_bboxes_neg_per_img[pos_inds]

            ref_ious = bbox_overlaps(pos_ref_bboxes_per_img, gt_bboxes_per_img_low)
            tgt_ious = bbox_overlaps(pos_tgt_bboxes_per_img, gt_bboxes_per_img_low)

            # refine assignment
            gt_inds_set = torch.unique(pos_assigned_gt_inds)
            mined_cnt = 0
            for gt_ind in gt_inds_set:
                pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
                target_labels = pos_labels[pos_idx_per_gt]
                ref_scores_per_gt = pos_ref_scores_per_img[pos_idx_per_gt, target_labels]
                tgt_scores_per_gt = pos_tgt_scores_per_img[pos_idx_per_gt, target_labels]
                tgt_ious_per_gt = tgt_ious[pos_idx_per_gt, gt_ind]
                ref_ious_per_gt = ref_ious[pos_idx_per_gt, gt_ind]

                gt_bboxes_per_gt = gt_bboxes_per_img_low[gt_ind:gt_ind + 1]
                tgt_bboxes_pos_per_gt = pos_tgt_bboxes_per_img[pos_idx_per_gt]

                diff_score = (tgt_scores_per_gt - ref_scores_per_gt)
                bboxes_ind = diff_score > self.train_cfg.diff_score_threshold

                # traceback to the original proposals indices
                indices = torch.arange(len(ref_bboxes[i]), dtype=gt_inds.dtype, device=gt_inds.device)
                mined_ind = indices[neg_inds][pos_inds][pos_idx_per_gt][bboxes_ind]

                if len(mined_ind) == 0:
                    continue
                # add mined results to the original assign results
                assign_result.gt_inds[mined_ind] = assign_result.num_gts + 1
                assign_result.num_gts += 1
                assign_result.labels[mined_ind] = target_labels[0]
                assign_result.max_overlaps[mined_ind] = ref_ious_per_gt[bboxes_ind]
                gt_bboxes_per_img_high = torch.cat([gt_bboxes_per_img_high, gt_bboxes_per_gt])
                gt_labels_per_img_high = torch.cat([gt_labels_per_img_high, target_labels[0:1]])
                mined_cnt += 1

            log_every_n({"mined_cls_bboxes": mined_cnt})
            gt_bboxes = list(gt_bboxes)
            gt_labels = list(gt_labels)
            gt_bboxes[i] = gt_bboxes_per_img_high
            gt_labels[i] = gt_labels_per_img_high

            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results, gt_bboxes, gt_labels

    def get_sampling_result_original(
            self,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            gt_bboxes_ignore=None,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results, gt_bboxes, gt_labels

    @torch.no_grad()
    def assignment_refinement(self, gt_inds, pos_inds, pos_assigned_gt_inds,
                              ious, cls_score, areas, labels):
        # (PLA) refine assignment results according to teacher predictions
        # on each image
        refined_gt_inds = gt_inds.new_full((gt_inds.shape[0],), -1)
        refined_pos_gt_inds = gt_inds.new_full((pos_inds.shape[0],), -1)

        gt_inds_set = torch.unique(pos_assigned_gt_inds)
        for gt_ind in gt_inds_set:
            # for cluster with class k `k=labels[pos_idx_per_gt]`,
            pos_idx_per_gt = torch.nonzero(pos_assigned_gt_inds == gt_ind).reshape(-1)
            target_labels = labels[pos_idx_per_gt]  # should be same for all proposals in cluster
            target_scores = cls_score[pos_idx_per_gt, target_labels]  # scores of class k for all proposals in cluster
            target_areas = areas[pos_idx_per_gt]  # areas for all proposals in cluster
            target_IoUs = ious[pos_idx_per_gt, gt_ind]  # ious wrt cluster center for all proposals in cluster

            cost = (target_IoUs * target_scores).sqrt()
            _, sort_idx = torch.sort(cost, descending=True)

            candidate_topk = min(pos_idx_per_gt.shape[0], self.PLA_candidate_topk)
            topk_ious, _ = torch.topk(target_IoUs, candidate_topk, dim=0)
            # calculate dynamic k for each gt
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
            sort_idx = sort_idx[:dynamic_ks]
            # filter some invalid (area == 0) proposals
            sort_idx = sort_idx[
                target_areas[sort_idx] > 0
                ]
            pos_idx_per_gt = pos_idx_per_gt[sort_idx]

            refined_pos_gt_inds[pos_idx_per_gt] = pos_assigned_gt_inds[pos_idx_per_gt]

        refined_gt_inds[pos_inds] = refined_pos_gt_inds
        return refined_gt_inds

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_proposal_prediction(self, feat, img_metas, proposal_list):
        bbox_preds_list, cls_scores_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, None, rescale=False
        )
        # decode all proposals bbox prediction
        decoded_bboxes_list = []
        for bbox_preds, cls_scores in zip(bbox_preds_list, cls_scores_list):
            if self.use_sigmoid:
                pred_labels = cls_scores.max(dim=-1)[1]
            else:
                # Note: Not the real prediction labels, just to get bboxes for the background prediction
                pred_labels = cls_scores[:, :self.num_classes - 1].max(dim=-1)[1]
            bbox_preds_ = bbox_preds.view(
                bbox_preds.size(0), -1,
                4)[torch.arange(bbox_preds.size(0)), pred_labels]
            decoded_bboxes_list.append(bbox_preds_)
        return cls_scores_list, bbox_preds_list, decoded_bboxes_list

    def extract_teacher_info(self, feat, img_metas):
        proposal_cfg = self.teacher.train_cfg.get(
            "rpn_proposal", self.teacher.test_cfg.rpn
        )
        rpn_out = list(self.teacher.rpn_head(feat))
        proposal_list = self.teacher.rpn_head.get_bboxes(
            *rpn_out, img_metas=img_metas, cfg=proposal_cfg
        )
        proposals = copy.deepcopy(proposal_list)

        # split RCNN prediction in to forward and  NMS step
        # proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
        #     feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        # )
        cls_scores_list, bbox_preds_list, decoded_bboxes_list = self.extract_proposal_prediction(
            feat, img_metas, proposal_list)

        proposal_list, proposal_label_list = [], []
        cfg = self.teacher.test_cfg.rcnn
        for bbox_preds, cls_scores in zip(bbox_preds_list, cls_scores_list):
            det_bboxes, det_labels = multiclass_nms(bbox_preds, cls_scores,
                                                    cfg.score_thr, cfg.nms, cfg.max_per_img)
            proposal_list.append(det_bboxes)
            proposal_label_list.append(det_labels)

        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        return det_bboxes, det_labels, proposals, feat, decoded_bboxes_list, cls_scores_list

    def compute_uncertainty_with_aug(
            self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                    torch.randn(times, box.shape[0], 4, device=box.device)
                    * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def convert_bbox_space(self, img_metas_A, img_metas_B, bboxes_A):
        """
            function: convert bboxes_A from space A into space B
            Parameters:
                img_metas: list(dict); bboxes_A: list(tensors)
        """
        transMat_A = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                      for meta in img_metas_A]
        transMat_B = [torch.from_numpy(meta["transform_matrix"]).float().to(bboxes_A[0].device)
                      for meta in img_metas_B]
        M = self._get_trans_mat(transMat_A, transMat_B)
        bboxes_B = self._transform_bbox(
            bboxes_A,
            M,
            [meta["img_shape"] for meta in img_metas_B],
        )
        return bboxes_B
