import cv2
import numpy as np
from agents.match_result import MatchResult

class Algorithm:
    def default_params(self):
        return {
            'method': 'ORB',
            'threshold': 0.7,
            'confidence_threshold': 0.3
        }

    def run(self, template, scene, **params):
        params = {**self.default_params(), **params}
        
        if template is None or scene is None:
            raise ValueError("Template and scene must be provided")
        
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(scene.shape) == 3:
            scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        
        try:
            method = params['method']
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(scene, None)
            
            if len(kp1) < 4 or len(kp2) < 4:
                return MatchResult(method, False, 0.0)
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            
            ratio_threshold = params.get('threshold', 0.7)
            good_matches = []
            for m_n in matches:
                if len(m_n) >= 2:
                    m, n = m_n[0], m_n[1]
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            confidence = min(1.0, len(good_matches) / max(len(kp1), 1))
            found = confidence >= params.get('confidence_threshold', 0.3)
            
            if found and len(good_matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = template.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    x = int(np.min(dst[:, :, 0]))
                    y = int(np.min(dst[:, :, 1]))
                    width = int(np.max(dst[:, :, 0]) - x)
                    height = int(np.max(dst[:, :, 1]) - y)
                else:
                    x, y = 0, 0
                    width, height = template.shape[1], template.shape[0]
            else:
                x, y = 0, 0
                width, height = template.shape[1], template.shape[0]
            
            result = MatchResult(
                algorithm=method,
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