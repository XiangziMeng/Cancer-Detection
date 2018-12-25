from utils import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3), padding='same')
        self.max_pooling1 = MaxPooling2D()
        self.conv_layer2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')
        self.max_pooling2 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense_layer1 = Dense(128, activation='relu')
        self.dense_layer2 = Dense(2)

    def call(self, x):
        x = self.conv_layer1(x)
        x = self.max_pooling1(x)
        x = self.conv_layer2(x)
        x = self.max_pooling2(x)   
        x = self.flatten(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        return x
    
def calculate_loss(logits, labels):
    m1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(m1)

def train(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss_value = calculate_loss(logits, labels)  
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))
    return loss_value

def train_model(train_generator, val_generator, optimizer, epochs, logdir='./log/exp'):
    model = MyModel()
    step_counter = 0
    
    writer = tf.contrib.summary.create_file_writer(logdir=logdir, flush_millis=1000)
    with writer.as_default():
        with tf.contrib.summary.always_record_summaries():
            for epoch_n in range(epochs):
                print('Epoch #%d' % (epoch_n))
                for images, labels in train_generator: 
                    images = tf.convert_to_tensor(images)
                    step_counter +=1                    
                    
                    loss = train(model, images, labels, optimizer)
                    tf.contrib.summary.scalar("loss", loss, step=step_counter)
                    
                    if step_counter % 10 == 0:
                        for test_images, test_labels in val_generator:
                            break
                        logits = model(test_images)
                        val_loss = calculate_loss(logits, test_labels)  
                        print('step: %d, loss: %.4f\tvalidation loss: %.4f' % (step_counter, loss, val_loss))
                        
                        tf.contrib.summary.scalar("val_loss", val_loss, step=step_counter)
                    
                    if step_counter % 100 == 0:
                        break
    writer.close()
    return model

def predict(model, label_file, save_file):
    fp = open(save_file, 'w')
    for line in open(label_file):
        image_path, label, x, y = line.strip().split('\t')
        image = imread(image_path) / 255.0
        score = model(np.array([image]))
        score = np.exp(score) / np.sum(np.exp(score))
        fp.write('%s\t%s\t%s\t%s\t%f\n' % (image_path, label, x, y, score[0][1]))
    return

def cal_recall_precision(prediction_file, threshold):
    tp, tn, fp, fn = 0, 0, 0, 0
    epsilon = 0.0001
    for line in open(prediction_file):
        image_path, label, x, y, score = line.strip().split('\t')
        score = float(score)
        if label == 'tumor' and score > threshold:
            tp += 1
        elif label == 'tumor' and score <= threshold:
            fn += 1
        elif label == 'normal' and score > threshold:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return precision, recall

def plot_pr(prediction_file, save_file):
    precisions, recalls = [], []
    for step in range(100):
        threshold = step * 0.01
        precision, recall = cal_recall_precision(prediction_file, threshold)
        precisions.append(precision)
        recalls.append(recall)
    
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(recalls, precisions)
    plt.savefig(save_file)
    plt.close()

if __name__ == '__main__':   
    data_folder = './data'
    level = 1
    train_generator = image_generator('%s/sampled_train/level_%d/label.txt' % (data_folder, level), 32)
    val_generator = image_generator('%s/val/level_%d/label.txt' % (data_folder, level), 32)
    

    print('Start training model')
    model = train_model(train_generator, val_generator, optimizer=tf.train.AdamOptimizer(0.001), epochs=5, logdir='./log/exp')
    
    print('Start making predictions on validating and testing data')
    predict(model, '%s/val/level/label.txt' % data_folder, './prediction/val_prediction_level_%d.txt' % level)
    predict(model, '%s/test_data/level/label.txt' % data_folder, './prediction/test_prediction_level_%d.txt' % level)
    
    print('Plot precision-recall curves on validating and testing data')
    plot_pr('./prediction/val_prediction_level_%d.txt' % level, './pr_curve/pr_val_level_%d.png' % level)
    plot_pr('./prediction/test_prediction_level_%d.txt' % level, './pr_curve/pr_test_level_%d.png' % level)
    
    print('Print table of precision, recall, threshold for choosing threshold')
    print('precision', 'recall', 'threshold')
    for step in range(20):
        threshold = step * 0.05
        precision, recall = cal_recall_precision('./prediction/val_prediction_level_%d.txt' % level, threshold)
        print(precision, recall, threshold)
    
