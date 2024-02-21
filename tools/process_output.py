import numpy as np
import json
import pandas as pd
import math
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats

def get_err(summary, indices, category='ETT'):
    with open(summary, "r") as f:
        data = json.load(f)
    with open(indices, "r") as fi:
        pixel_spacing = json.load(fi)
    euc_dis = []
    y_dis = []
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in data.keys():
        gt = data[i][category]['GT']
        pred = data[i][category]['pred']
        if any(math.isnan(t) for t in gt):
            if any(math.isnan(t) for t in pred):
                tn += 1
            else:
                fp += 1
        elif any(math.isnan(t) for t in pred):
            fn += 1
        else:
            tp += 1
            d = math.sqrt((gt[0]-pred[0])**2+(gt[1]-pred[1])**2) * pixel_spacing[i]['pixel_spacing']/10
            
            if d == None:
                print(gt)
                print(pred)
            euc_dis.append(d)
            y_dis.append((pred[1] - gt[1])* pixel_spacing[i]['pixel_spacing']/10)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    print(f"Mean error: {np.mean(euc_dis):.2f}")
    print(f"Median error: {np.median(euc_dis):.2f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")
    # indices_sorted = sorted(range(len(euc_dis)), key=lambda i: euc_dis[i], reverse=True)
    # print(indices_sorted)
    return euc_dis, y_dis

def carina_net_err(output_file, indices, category='ETT'):
    with open(output_file, "r") as f:
        data = json.load(f)
    with open(indices, "r") as fi:
        pixel_spacing = json.load(fi)
    dis = []
    for i in data.keys():
        gt = data[i][category]['GT']
        pred = data[i][category]['pred']
        d = (pred[1]-gt[1]) * pixel_spacing[i]['pixel_spacing']/10
        dis.append(abs(d))
    # print(dis)
    print(f"Mean: {np.nanmean(dis)}")
    print(f"Median: {np.nanmedian(dis)}")
    return dis

    
def plot_dis(data, hospital, title, outfile):
    # Create a histogram
    fig, ax = plt.subplots()
    ax.hist(data, bins=20, edgecolor='k', alpha=0.7)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{hospital} {title}')

    # Calculate quantiles
    q25 = np.nanpercentile(data, 25)
    q75 = np.nanpercentile(data, 75)

    # Display quantiles
    ax.axvline(q25, color='r', linestyle='dashed', linewidth=2, label='25th Quantile')
    ax.axvline(q75, color='g', linestyle='dashed', linewidth=2, label='75th Quantile')
    ax.legend()

    fig.savefig(f'/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/outputs/{hospital}/{outfile}.png', dpi=300, bbox_inches='tight')

def plot_cumulative_err(df, hospital):
    # Sort the DataFrame by the 'tip-carina error' column
    df = df.sort_values(by='Tip-Carina Error')

    # Create a cumulative count of items less than or equal to each error
    cumulative_count = [i + 1 for i in range(len(df))]

    fig, ax = plt.subplots()

    # Plot the cumulative count against the error values
    ax.plot(df['Tip-Carina Error'], cumulative_count)

    # Set labels and title
    ax.set_xlabel('Error Value')
    ax.set_ylabel('Cumulative Count')
    ax.set_title(f'{hospital} Abs. T-C Error between GT and Pred')

    # Show the plot
    fig.savefig(f'/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/outputs/{hospital}/cum_err.png')

def get_spread_sheet(hospital, summary, indices):
    with open(summary, "r") as f:
        data = json.load(f)
    with open(indices, "r") as f:
        indices = json.load(f)
    output = []
    for i in data.keys():
        image_name = indices[i]["path"].split('/')[-1]
        gt_ETT = data[i]['ETT']['GT']
        pred_ETT = data[i]['ETT']['pred']
        gt_carina = data[i]['CARINA']['GT']
        pred_carina = data[i]['CARINA']['pred']
        
        tip_err, carina_err, gt_tip_carina_err, pred_tip_carina_err, tip_carina_err = None, None, None, None, None
        tip, carina = False, False
        if not any(math.isnan(t) for t in gt_ETT) and not any(math.isnan(t) for t in pred_ETT):
            tip = True
            tip_err = math.sqrt((gt_ETT[0]-pred_ETT[0])**2+(gt_ETT[1]-pred_ETT[1])**2) * indices[i]['pixel_spacing']/10
        if not any(math.isnan(t) for t in gt_carina) and not any(math.isnan(t) for t in pred_carina):
            carina = True
            carina_err = math.sqrt((gt_carina[0]-pred_carina[0])**2+(gt_carina[1]-pred_carina[1])**2) * indices[i]['pixel_spacing']/10
        if tip and carina:
            gt_tip_carina_err = math.sqrt((gt_ETT[0]-gt_carina[0])**2+(gt_ETT[1]-gt_carina[1])**2) * indices[i]['pixel_spacing']/10
            pred_tip_carina_err = math.sqrt((pred_ETT[0]-pred_carina[0])**2+(pred_ETT[1]-pred_carina[1])**2) * indices[i]['pixel_spacing']/10
            tip_carina_err = abs(gt_tip_carina_err - pred_tip_carina_err)
        output.append([i, image_name, gt_ETT, pred_ETT, gt_carina, pred_carina,
                     tip_err, carina_err, gt_tip_carina_err, pred_tip_carina_err, tip_carina_err])
    output = pd.DataFrame(output, columns=['Index', 'Image', 'GT ETT', 'Pred ETT', 
        'GT Carina', 'Pred Carina', 'Tip Prediction Error',
        'Carina Prediction Error', 'GT Tip-Carina Distance',
        'Pred Tip-Carina Distance', 'Tip-Carina Error'])
    return output
    
        

    

if __name__ == "__main__":
    hospitals = ['Ascension-Seton', 'Cedars-Sinai','Chiang_Mai_University', 
                 'Fundación_Santa_Fe_de_Bogotá', 'Lawson_Health', 
                 'Morales_Meseguer_Hospital', 'National_University_of_Singapore',
                 'Newark_Beth_Israel_Medical_Center', 'NYU_Langone_Health',
                 'Osaka_City_University', 'Rhode_Island_Hospital', 
                 'Sunnybrook_Research_Institute', 'Technical_University_of_Munich',
                 'Universitätsklinikum_Essen', 'Universitätsklinikum_Tübingen', 
                 'University_of_Miami']
    fig, ax = plt.subplots()
    y_dis_all = []
    for hospital in hospitals:
        print(hospital)
        summary = f'/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/outputs/{hospital}/CarinaNet/CarinaNet_summary.json' 
        indices = f'/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/outputs/{hospital}/indices.json'
        
        output = get_spread_sheet(hospital, summary, indices)
        # plot_cumulative_err(output, hospital)
        # output.to_csv(f'/n/scratch3/users/c/cat302/ETT-Project/CarinaNet/outputs/{hospital}/err_report.csv', index=False)
        
        ## Euclidean error
        # print("CARINA")
        # get_err(summary, indices, category='CARINA')
        # print("ETT")
        # euc_dis, y_dis = get_err(summary, indices, category='ETT')
        # plot_dis(euc_dis, hospital, title='Histogram of Euclidean Tip Error',
        #          outfile='histogram')
        # plot_dis(y_dis, hospital, title='Histogram of Tip Position Error (Pred - GT)',
        #          outfile='y_err')


        ## Test for CarinaNet output
        # print("CARINA")
        # carina_net_err(summary, indices, category='CARINA')
        # print("ETT")
        # dis = carina_net_err(summary, indices, category='ETT')
        ax.hist(output['Tip-Carina Error'], bins=20, edgecolor='k', alpha=0.3, label=hospital)
        # y_dis_all.extend(y_dis)
    ax.legend()
    ax.set_xlabel("Tip-Carina Error")
    ax.set_ylabel("Frequency")
    ax.set_title("CarinaNet")
    fig.savefig("results/overall_t-c_err.png")
    
    # print(y_dis_all)
    # # Perform the Shapiro-Wilk test
    # statistic, p_value = stats.shapiro(y_dis_all)

    # # Check if the data follows a normal distribution
    # alpha = 0.05  # Significance level
    # print(p_value)
    # print(statistic)
    # if p_value > alpha:
    #     print("Data appears to be normally distributed")
    # else:
    #     print("Data does not appear to be normally distributed")
        