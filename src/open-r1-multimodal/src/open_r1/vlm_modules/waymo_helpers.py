from collections import defaultdict
import numpy as np
from open_r1.vlm_modules.rater_feedback_utils import get_rater_feedback_score
import pickle
from scipy.interpolate import CubicSpline  # TODO: vlm-r1 env needs scipy
from torch.utils.data import Dataset


def compute_rater_feedback_scores(prediction,
                                  scenario_id,
                                  seq):
    _, trajectories, initial_speed, scores = preference_dict[f"{scenario_id}-{seq}"]

    rater_specified_trajectories = [trajectories]
    initial_speed_list = np.array([initial_speed])
    rater_scores_list = [scores]
    prediction_trajectories = np.array([prediction])
    prediction_probabilities = np.array([1])

    rater_feedback_metrics = (
        get_rater_feedback_score(
            prediction_trajectories,
            prediction_probabilities,
            rater_specified_trajectories,
            rater_scores_list,
            initial_speed_list,
            frequency=4,  # Default is 4.
            length_seconds=5,  # Default predict 5 seconds.
            output_trust_region_visualization=False,
        )
    )

    rfs = rater_feedback_metrics['rater_feedback_score']

    return rfs


def get_interp_pred(pred):
    t_src = np.arange(0, 6)  # 0…5 s
    t_tgt = np.arange(0.25, 5.01, 0.25)  # 0.25…5 s

    pred_with_t0 = np.zeros((6, 2))
    pred_with_t0[1:] = pred
    pred = pred_with_t0
    cs_x = CubicSpline(t_src, pred[:, 0], bc_type='natural')
    cs_y = CubicSpline(t_src, pred[:, 1], bc_type='natural')
    xy_dense = np.stack([cs_x(t_tgt), cs_y(t_tgt)], axis=-1).round(2)  # (20,2)

    pred = xy_dense

    return pred


def compute_ade(prediction, reference):
    """
    Computes the Average Displacement Error (ADE) and Final Displacement Error (FDE)
    between the predicted trajectory and the ground truth trajectory.

    Args:
        prediction (np.ndarray): Predicted trajectory of shape (N, 2) where N is the number of time steps.
        reference (np.ndarray): Reference trajectory of shape (N, 2).

    Returns:
        tuple: ADE
    """
    ade = np.mean(np.linalg.norm(get_interp_pred(prediction[3::4]) - reference, axis=1))

    return ade
