import time

import matplotlib.pyplot as plt
import numpy as np

from Dataset import dataset
import math

def _uncertainty_quantification(index, carinaNet_summary):
    """Uses a correlation between the confidence and the average error"""
    min_confidence = min(carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence'])
    return 4.3*np.exp(-3.3*min_confidence)+0.1 #average error in cm

def _alter_image(img, points, ETT, carina, gt_ETT, gt_carina, uncertainty, index):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img, cmap = 'gray')
    ax.scatter(*zip(*points[:,::-1]), alpha = 0.1, c = 'r', s = 3, label = 'Predicted Endotracheal tube')
    ax.scatter(*ETT, c = 'purple', s = 5, alpha = 1, label = 'Predicted Endotracheal tube tip')
    ax.scatter(*carina, c = 'green', s = 5, alpha = 1, label = 'Predicted Carina')
    ax.scatter(*gt_ETT, c = 'pink', s = 5, alpha = 1, label = 'GT Endotracheal tube tip')
    ax.scatter(*gt_carina, c = 'blue', s = 5, alpha = 1, label = 'GT Carina')
    # dist = dataset.metrics.to_cm(index) * (carina[1]-ETT[1])
    # ax.annotate(f'ETT-carina distance = {dist:.1f}cm $\pm$ {uncertainty:.1f}cm',
    #             xy=(0.2, 0.05), xycoords='axes fraction', fontsize=8,
    #             bbox=dict(facecolor='none', edgecolor='k', pad=3), color = 'k')
    gt_ett_carina_dis = dataset.metrics.to_cm(index) * math.sqrt((gt_carina[0]-gt_ETT[0])**2+(gt_carina[1]-gt_ETT[1])**2)
    pred_ett_carina_dis = dataset.metrics.to_cm(index) * math.sqrt((carina[0]-ETT[0])**2+(carina[1]-ETT[1])**2)
    ax.annotate(f'GT ETT-carina distance = {gt_ett_carina_dis:.2f}cm.\nPredicted ETT-carina distance = {pred_ett_carina_dis:.2f}cm',
                xy=(0.2, 0.05), xycoords='axes fraction', fontsize=8,
                bbox=dict(facecolor='none', edgecolor='k', pad=3), color = 'k')
    leg = ax.legend(prop = {'size': 5})
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    fig.savefig(dataset.paths.augmented_image(index), bbox_inches="tight", pad_inches=0,
                dpi = 500)


def main(indices):
    t0 = time.time()
    print('   CREATING OUTPUT')
    carinaNet_summary = dataset.summaries.load('CarinaNet', 'CarinaNet')
    for index in indices:
        img = dataset.load.image(index)
        points = dataset.load.ETT_detection(index)
        if points is None:
            continue
        ETT = carinaNet_summary[index]['ETT']['pred']
        carina = carinaNet_summary[index]['CARINA']['pred']
        gt_ETT = carinaNet_summary[index]['ETT']['GT']
        gt_carina = carinaNet_summary[index]['CARINA']['GT']
        uncertainty = _uncertainty_quantification(index, carinaNet_summary)
        _alter_image(img, points, ETT, carina, gt_ETT, gt_carina, uncertainty, index)

    print(f'      -> Done in :{time.time() - t0:.2f}s\n')
