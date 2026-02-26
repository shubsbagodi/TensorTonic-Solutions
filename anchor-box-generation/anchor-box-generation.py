import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generates anchor boxes for an object detector on a feature grid.
    """
    # 1. Compute the stride (spacing between grid cells in image space)
    stride = image_size / feature_size
    
    anchors = []
    
    # 2. Iterate over grid cells in row-major order (i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            
            # Compute the center in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # 3. For each cell, iterate over scales then aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    
                    # Compute box width and height
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # Calculate corners
                    x1 = cx - (w / 2.0)
                    y1 = cy - (h / 2.0)
                    x2 = cx + (w / 2.0)
                    y2 = cy + (h / 2.0)
                    
                    anchors.append([x1, y1, x2, y2])
                    
    return anchors