import cv2
import numpy as np
from agents.match_result import MatchResult

class Algorithm:
    def default_params(self):
        return {
            'method': 'HISTCMP_CORREL',
            'confidence_threshold': 0.3
        }

    def run(self, template, scene, **params):
        params = {**self.default_params(), **params}
        
        if template is None or scene is None:
            raise ValueError("Template and scene must be provided")
        
        # Convert to grayscale
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(scene.shape) == 3:
            scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        
        try:
            method_str = params['method']
            if method_str == 'HISTCMP_CORREL':
                method = cv2.HISTCMP_CORREL
            elif method_str == 'HISTCMP_CHISQR':
                method = cv2.HISTCMP_CHISQR
            elif method_str == 'HISTCMP_INTERSECT':
                method = cv2.HISTCMP_INTERSECT
            elif method_str == 'HISTCMP_BHATTACHARYYA':
                method = cv2.HISTCMP_BHATTACHARYYA
            else:
                method = cv2.HISTCMP_CORREL
            
            hist_template = cv2.calcHist([template], [0], None, [256], [0, 256])
            hist_scene = cv2.calcHist([scene], [0], None, [256], [0, 256])
            
            hist_template = cv2.normalize(hist_template, hist_template).flatten()
            hist_scene = cv2.normalize(hist_scene, hist_scene).flatten()
            
            confidence = cv2.compareHist(hist_template, hist_scene, method)
            
            h, w = template.shape[:2]
            res = cv2.matchTemplate(scene, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            x, y = top_left
            width, height = w, h
            
            found = confidence >= params.get('confidence_threshold', 0.3)
            
            result = MatchResult(
                algorithm=params['method'],
                found=found,
                confidence=float(confidence),
                location=(x, y),
                size=(width, height)
            )
            result.params = params
            return result
            
        except Exception as e:
            return MatchResult(
                algorithm=params.get('method', 'unknown'),
                found=False,
                confidence=0.0
            )