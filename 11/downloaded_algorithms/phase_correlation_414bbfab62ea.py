import cv2
import numpy as np
from agents.match_result import MatchResult

class Algorithm:
    def default_params(self):
        return {
            'method': 'PHASE_CORRELATION'
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
            h, w = template.shape[:2]
            
            # Perform phase correlation
            shift_map = cv2.phaseCorrelate(scene, template)
            y, x = map(int, shift_map)
            
            # Calculate confidence
            corr_value = np.abs(cv2.phaseCorrelate(scene, template)[0])
            confidence = min(1.0, corr_value)
            
            # Determine location and size
            x_min, y_min = max(x, 0), max(y, 0)
            x_max, y_max = min(x + w, scene.shape[1]), min(y + h, scene.shape[0])
            width, height = x_max - x_min, y_max - y_min
            
            result = MatchResult(
                algorithm=params['method'],
                found=True,
                confidence=float(confidence),
                location=(x_min, y_min),
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