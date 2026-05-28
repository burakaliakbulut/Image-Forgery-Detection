import cv2
import numpy as np
from PIL import Image

def _load_image(image_path):
    """Görseli PIL ile güvenli bir şekilde okur ve OpenCV formatına çevirir."""
    try:
        pil_img = Image.open(image_path).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def _get_detector(algorithm):
    """Seçilen algoritmaya göre uygun dedektörü döndürür."""
    if algorithm == "SIFT":
        return cv2.SIFT_create()
    if algorithm == "SURF":
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    if algorithm == "AKAZE":
        return cv2.AKAZE_create()
    if algorithm == "ORB":
        return cv2.ORB_create(nfeatures=1000)
    
    raise ValueError("Desteklenmeyen bir algoritma girdiniz.")

def _filter_good_matches(matches, keypoints):
    """Lowe's Ratio ve mesafe testinden geçen kaliteli eşleşmeleri süzer."""
    good_matches = []
    
    for match in matches:
        # Guard Clause: Yeterli eşleşme yoksa doğrudan atla (İç içe if'ten kurtulduk)
        if len(match) < 3:
            continue
            
        _, n, o = match
        
        # Guard Clause: Kalite farkı yeterli değilse atla
        if n.distance >= 0.75 * o.distance:
            continue
            
        pt1 = np.array(keypoints[n.queryIdx].pt)
        pt2 = np.array(keypoints[n.trainIdx].pt)
        
        # Orijinal parça ile kopya parça fiziksel olarak en az 40 piksel uzak olmalı
        if np.linalg.norm(pt1 - pt2) > 40:
            good_matches.append((pt1, pt2))
            
    return good_matches

def detect_copy_move_forgery(image_path, algorithm="SIFT"):
    """Ana fonksiyon: Resmi okur, anahtar noktaları çıkarır ve eşleştirir."""
    img = _load_image(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = _get_detector(algorithm)

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None or len(descriptors) < 3:
        return None

    # Algoritmaya göre eşleştirici normunu seç
    norm_type = cv2.NORM_HAMMING if algorithm == "ORB" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
        
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    good_matches = _filter_good_matches(matches, keypoints)

    if not good_matches:
        return None

    # Çizim işlemleri
    result_img = img.copy()
    for pt1, pt2 in good_matches:
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        
        cv2.line(result_img, p1, p2, (0, 255, 0), 2)
        cv2.circle(result_img, p1, 5, (0, 0, 255), -1)
        cv2.circle(result_img, p2, 5, (255, 0, 0), -1)

    return result_img