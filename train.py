import argparse
from sys import argv
from os import listdir
from skimage.io import imread
from os.path import join
from crop import getCropRect, findCircle
from summary import getPixelSummary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def filterImage(filename):
    extension = filename.split('.')[-1]
    if extension == 'jpg' or extension == 'jpeg' or extension == 'png' or extension == 'tif':
        return True
    return False

def prepareData(train_directory, label_file):
    train_images = list(filter(filterImage, listdir(train_directory)))
    train_dict = {}
    for image in train_images:
        rect, _ = getCropRect(imread(join(train_directory, image)))
        circles = findCircle(rect)
        if len(circles) == 3:
            train_dict[image] = { 'image': rect, 'circles': circles }
    label = pd.read_csv(label_file)
    summary_df = pd.DataFrame(getPixelSummary(train_dict, label))
    return summary_df

def extractFeature(summary):
    dataset = []
    label = []
    for index, row in summary.iterrows():
        each_label = row["label"]
        background_avg_r = row['background_avg_r']
        background_avg_g = row['background_avg_g']
        background_avg_b = row['background_avg_b']
        background_std_r = row['background_std_r']
        background_std_g = row['background_std_g']
        background_std_b = row['background_std_b']
        for i in ['1','2','3']:
            r = row['avg_r'+i]
            g = row['avg_g'+i]
            b = row['avg_b'+i]
            std_r = row['std_r'+i]
            std_g = row['std_g'+i]
            std_b = row['std_b'+i]
            dataset.append([
                r-background_avg_r,
                g-background_avg_g,
                b-background_avg_b,
            ])
            label.append(each_label)
    return dataset, label

def train(summary_df):
    summary = summary_df.dropna()
    dataset, label = extractFeature(summary)
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.1, random_state=1)
    model = LinearRegression()
    model.fit(X_train,y_train)
    print('R^2 score on train: {}'.format(model.score(X_train, y_train)))
    print('R^2 score on test: {}'.format(model.score(X_test, y_test)))
    return model

def main():
    parser = argparse.ArgumentParser(description="G6PD Prediction model training script.")
    parser.add_argument('-t', '--train', help='Path to train set directory.', required=True)
    parser.add_argument('-l', '--label', help='Path to label csv file', required=True)
    # parser.add_argument('-o', '--output', help='Training output file', required=True)
    args = parser.parse_args(argv[1:])

    summary_df = prepareData(args.train, args.label)
    model = train(summary_df)
    print('coef: {}'.format(model.coef_))
    print('intercept: {}'.format(model.intercept_))

if __name__ == "__main__":
    main()