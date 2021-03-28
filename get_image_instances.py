import os
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from instance_identifier.instance_identifier import InstanceIdentifier


def get_all_images(images_folder):
    xx = []
    yy = []
    for fld in os.listdir(images_folder):
        fld_path = images_folder + fld + '/'

        for img in os.listdir(fld_path):
            img_path = fld_path + img
            xx.append(img_path)
            yy.append(fld)
    return xx, yy


def main():
    train_images_folder = './data/embedding_encoder/training/train/'
    val_images_folder = './data/embedding_encoder/training/val/'

    ii = InstanceIdentifier()

    x_train, y_train = get_all_images(train_images_folder)
    x_test, y_test = get_all_images(val_images_folder)

    x_train = ii.extract_all_embeddings(x_train)
    print()
    x_test = ii.extract_all_embeddings(x_test)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_pred = neigh.predict(x_test)

    return classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    report, cfs_matrix = main()
    print(report)
    plt.imshow(cfs_matrix)
    plt.show()
