import skimage
import numpy as np
import matplotlib.pyplot as plt
import skfmm
import cv2

def norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def sdf(mask, organ=1):
    
    phi = np.zeros((1024,1024))
    phi[mask == organ] = 1

    edges = skimage.feature.canny(
        image=phi,
        sigma=2,
        low_threshold=0.5,
        high_threshold=1.5,
    )

    edges = edges.astype(np.float32)

    phi = np.where(edges, 0, 1)

    sd = skfmm.distance(phi, dx = .1)

    #print(np.max(sd), np.min(sd))

    sd = norm(sd)
    #sd = np.clip(sd, 0, 0.4)

    #print(np.max(sd), np.min(sd))

    # Plot results
    #plt.figure(figsize=(20,20))
    #plt.imshow(sd)
    ##plt.colorbar()
    #plt.axis('off')
    #plt.savefig('Evaluation/Graphics/Heart_SDF.png', dpi=300)
    
    return sd