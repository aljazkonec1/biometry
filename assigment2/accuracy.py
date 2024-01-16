import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import glob
from skimage.feature import match_descriptors, SIFT, plot_matches

def calculate_similarity(incoming_img, database_images):

    incoming_img_feature = np.loadtxt(incoming_img).reshape(-1,1 ).T
    
    zac = np.loadtxt(database_images[0]).reshape(-1,1 ).T
    best_sim = abs(cosine_similarity(incoming_img_feature,zac )[0][0])

    # best_sim = euclidean_distances(incoming_img_feature, zac)[0][0]

    best_match = os.path.splitext(os.path.basename(database_images[0]))[0].split("-")[0]

    for db_ear in database_images:
            ear_feature = np.loadtxt(db_ear).reshape(-1,1 ).T

            sim = abs(cosine_similarity(incoming_img_feature, ear_feature)[0][0])
            # sim = euclidean_distances(incoming_img_feature, ear_feature)[0][0]

            if sim >= best_sim:
                best_sim = sim
                best_match = os.path.splitext(os.path.basename(db_ear))[0].split("-")[0]
                best_ear = db_ear
                print(best_sim, best_match, os.path.basename(incoming_img))

    return sim, best_match



def calculate_sift_similarity(incoming_img, database_images):
         
        incoming_img_feature = np.loadtxt(incoming_img, delimiter=",", dtype=np.float64)
        zac = np.loadtxt(database_images[0], delimiter=",", dtype=np.float64)
        best_sim = 0
        
        matches = match_descriptors(incoming_img_feature, zac, max_ratio=0.6,cross_check=True)
        for m in matches:
             best_sim = best_sim + abs(cosine_similarity(incoming_img_feature[m[0], :].reshape(-1,1 ),zac[m[1], :].reshape(-1,1 ))[0][0])

        best_match = os.path.splitext(os.path.basename(database_images[0]))[0].split("-")[0]
    
        for db_ear in database_images:
                ear_feature = np.loadtxt(db_ear, delimiter=",", dtype=np.float64)
                
                sim = 0
                matches = match_descriptors(incoming_img_feature, ear_feature, max_ratio=0.6,cross_check=True)
                for m in matches:
                    des1 = incoming_img_feature[m[0], :].reshape(1, -1)
                    des2 = ear_feature[m[1], :].reshape(1, -1)

                    sim = sim + abs(cosine_similarity(des1, des2)[0][0])
    
                if sim > best_sim:
                    best_sim = sim
                    best_match = os.path.splitext(os.path.basename(db_ear))[0].split("-")[0]
                    best_ear = db_ear
                    print(best_sim, best_match, os.path.basename(incoming_img))
    
        return sim, best_match

def calculate_accuracy(matches):
    correct = 0
    for match in matches:
        if match[0] == match[1]:
            correct += 1
    return correct



options = ['test_lbp', 'test_resnet', 'test_sk_lbp', 'test_resnet_3']
# options = ['test_resnet_3']

path = os.path.join(os.getcwd(), 'datasets', 'ears', 'images-cropped')

stats = {}

for opt in options:
    p = os.path.join(path, opt)
    images = glob.glob(os.path.join(p, '*.csv'))
    matches = []

    for incoming_img in images:
        database_images = glob.glob(os.path.join(p, '*.csv'))
        database_images.remove(incoming_img)

        img_name = os.path.splitext(os.path.basename(incoming_img))[0]
        img_id = img_name.split("-")[0]

        similarity, best_match = calculate_similarity(incoming_img, database_images)
        matches.append([img_id, best_match, similarity])
    
    matches = np.array(matches)

    c = calculate_accuracy(matches)
    stats[opt] = {'Correct number':c , 'Total': len(matches), 'Accuracy': c/len(matches)}


p = os.path.join(path, 'test_sift')
images = glob.glob(os.path.join(p, '*.csv'))
matches = []

for incoming_img in images:
    database_images = glob.glob(os.path.join(p, '*.csv'))
    database_images.remove(incoming_img)

    img_name = os.path.splitext(os.path.basename(incoming_img))[0]
    img_id = img_name.split("-")[0]

    similarity, best_match = calculate_sift_similarity(incoming_img, database_images)
    matches.append([img_id, best_match, similarity])

matches = np.array(matches)

c = calculate_accuracy(matches)
stats['test_sift'] = {'Correct number':c , 'Total': len(matches), 'Accuracy': c/len(matches)}



print(stats)
df = pd.DataFrame(stats)
print(df)
df.to_csv(path + '/accuracy_sift.csv')

