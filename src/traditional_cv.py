import cv2
import numpy as np
from PIL import Image

def detect_copy_move_forgery(image_path, algorithm="SIFT"):
    try:
        # GÜÇLÜ OKUMA YÖNTEMİ
        pil_img = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if algorithm == "SIFT":
        detector = cv2.SIFT_create()
    elif algorithm == "SURF":
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    elif algorithm == "AKAZE":
        detector = cv2.AKAZE_create()
    elif algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=1000)
    else:
        raise ValueError("Desteklenmeyen bir algoritma girdiniz.")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None or len(descriptors) < 3:
        return None

    if algorithm == "ORB":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
    # HATA BURADAYDI: Görüntüyü kendiyle eşleştirdiğimiz için k=3 yapmalıyız.
    # 1. eşleşme noktanın kendisi (çöpe atılacak), 2. eşleşme kopyası, 3. eşleşme alakasız benzer nokta.
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    good_matches = []
    for match in matches:
        # Eğer en az 3 eşleşme dönebilmişse (resim yeterince büyükse)
        if len(match) >= 3:
            m, n, o = match
            
            # n (Kopya nokta) ile o (Alakasız nokta) arasındaki kalite farkını ölçüyoruz
            if n.distance < 0.75 * o.distance:
                pt1 = np.array(keypoints[n.queryIdx].pt)
                pt2 = np.array(keypoints[n.trainIdx].pt)
                
                # Orijinal parça ile kopya parça fiziksel olarak en az 40 piksel uzak olmalı
                if np.linalg.norm(pt1 - pt2) > 40:
                    good_matches.append((pt1, pt2))

    if len(good_matches) == 0:
        return None

    result_img = img.copy()
    for pt1, pt2 in good_matches:
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        
        # Orijinal nesne ile kopyası arasına yeşil çizgi çek
        cv2.line(result_img, p1, p2, (0, 255, 0), 2)
        cv2.circle(result_img, p1, 5, (0, 0, 255), -1)
        cv2.circle(result_img, p2, 5, (255, 0, 0), -1)

    return result_img