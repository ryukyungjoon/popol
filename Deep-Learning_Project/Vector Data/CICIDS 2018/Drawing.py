import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Drawing:
    def print_confusion_matrix(confusion_matrix, class_names, normalize=None, figsize=(15, 15), fontsize=15):
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
        plt.figure(figsize=figsize)
        if normalize:
            df_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            print("Normalized confusion matrix")
        else:
            fmt = 'd'
            print('Confusion matrix, without normalization')

        try:
            heatmap = sns.heatmap(df_cm, cmap='Blues', annot=True, fmt=fmt, robust=True,
                                  linewidths=.5, annot_kws={"size": 15})
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def loss_graph(autoencoder_train):
        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Autoencoder Training and validation loss')
        plt.legend()
        plt.show()

    def loss_acc_graph(classify_train):
        accuracy = classify_train.history['acc']
        val_accuracy = classify_train.history['val_acc']
        loss = classify_train.history['loss']
        val_loss = classify_train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Full Model Training and validation loss')
        plt.legend()
        plt.show()
