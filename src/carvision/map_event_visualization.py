"""Diagnostics for temporal map changes and replanning decisions."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .map_events import MapChangeSet, RouteContext, TriggerDecision
from .mapping import FREE, OCCUPIED, UNKNOWN


def render_event_frame(change: MapChangeSet, context: RouteContext,
                       decision: TriggerDecision, *, scale: int = 5) -> NDArray[np.uint8]:
    data = change.stable
    image = np.zeros((*data.shape, 3), np.uint8)
    image[data == FREE] = (238, 238, 238)
    image[data == UNKNOWN] = (125, 125, 125)
    image[data == OCCUPIED] = (20, 20, 20)
    image[change.raw_changed] = (200, 80, 200)
    image[change.stable_changed] = (40, 210, 40)
    relevant = change.stable_changed & (context.safety_corridor | context.candidate_mask
                                        | context.divergence_mask | context.inspection_mask)
    image[relevant] = (0, 220, 255)
    image[change.raw_newly_occupied & context.stopping_corridor] = (0, 0, 255)
    image = cv2.resize(np.flipud(image), (data.shape[1] * scale, data.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    points = np.asarray([(column * scale + scale // 2,
                          (data.shape[0] - 1 - row) * scale + scale // 2)
                         for row, column in context.selected_path], np.int32)
    cv2.polylines(image, [points], False, (255, 255, 0), max(2, scale // 2), cv2.LINE_AA)
    panel = np.full((64, image.shape[1], 3), 250, np.uint8)
    cv2.putText(panel, f"frame={change.frame_index} action={decision.action.value}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, decision.reasons[0][:85], (6, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, "raw=purple stable=green relevant=yellow emergency=red path=cyan",
                (6, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack((panel, image))
