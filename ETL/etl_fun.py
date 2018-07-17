import pandas as pd 
import numpy as np
import os 
import matplotlib.pyplot as plt 
from shapely.wkt import loads
from matplotlib.patches import Polygon
from scipy.stats import pearsonr

DATA_DIR = os.path.join('~','.kaggle','competitions','dstl-satellite-imagery-feature-detection')
TRAIN_FILE = os.path.join( DATA_DIR, 'train_wkt_v4.csv')

def load_dataset( file ):
    df = pd.read_csv( file )  
    df['Polygons'] = df.apply(lambda row: loads(row.MultipolygonWKT), axis=1)
    return df  

def polygons_in_image( image_str, df ):
    image = df[df.ImageId == image_str]
    polygons = {}
    for class_t in image.ClassType.unique():
        polygons[class_t] = loads(image[image.ClassType == class_t].MultipolygonWKT.values[0])
    return polygons 

if __name__ == "__main__":
    df = pd.read_csv( TRAIN_FILE )

    # First image file
    polygonsList = {}
    image = df[df.ImageId == '6100_1_3']
    for class_type in image.ClassType.unique():
        polygonsList[class_type] = loads(image[image.ClassType == class_type].MultipolygonWKT.values[0])

    # Plot the first image
    fig, ax = plt.subplots(figsize=(8,8))
    for class_t in polygonsList:
        for polygons in polygonsList[class_t]:
            mpl_polygon = Polygon(np.array(polygons.exterior), color=plt.cm.Set1(class_t*11), lw=0, alpha=0.65)
            ax.add_patch(mpl_polygon)
    ax.relim()
    ax.autoscale_view()

    # plt.show()


    classes = df.ImageId.unique()
    print("{:} images in dataset".format(len(classes)))

    #Augement the stored dataframe 
    df['Polygons'] = df.apply(lambda row: loads(row.MultipolygonWKT), axis=1)
    df['nPolygons'] = df.apply(lambda row: len(row['Polygons'].geoms), axis=1)

    pivot = df.pivot(index="ImageId", columns="ClassType", values="nPolygons")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_aspect('equal')
    plt.imshow(pivot.T, interpolation='nearest', cmap=plt.cm.Reds, extent=[0,22,10,1])
    plt.yticks(np.arange(1,11,1))
    plt.title("Number of objects for class per image")
    plt.ylabel("Class type")
    plt.xlabel("Image")
    plt.colorbar()
    # plt.show()
    fig.savefig("./../pride/object_distribution_class.png")

    pvt = pivot
    print("Trees vs Buildings: {:5.4f}".format(pearsonr(pvt[1],pvt[5])[0]))
    print("Trees vs Buildings and Structures: {:5.4f}".format(pearsonr(pvt[1]+pvt[2],pvt[5])[0]))

    for img in df.ImageId.unique():
        polygons = polygons_in_image(img,df)
        fig, ax = plt.subplots(figsize=(8,8))
        for class_t in polygons:
            for polygon in polygons[class_t]:
                mpl_poly = Polygon(np.array(polygon.exterior),color=plt.cm.Set1(class_t*11),lw=0,alpha=0.65)
                ax.add_patch(mpl_poly)
        ax.relim()
        ax.autoscale_view()
        fig.suptitle(img)
        fig.savefig("./../pride/polygons_" + img + ".png")
        fig.gcf()
