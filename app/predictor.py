import numpy as np
from PIL import Image
import tensorflow as tf


class Predictor:
    def __init__(self, labels_path: str, model_path: str, image_size: int):
        self.labels_path = labels_path
        self.model_path = model_path
        self.image_size = image_size

        with open(self.labels_path, "r") as f:
            self.labels = [line.strip() for line in f]
        
        self.model = tf.keras.models.load_model(self.model_path)
    
    def predict_image(self, image: np.ndarray, n_top: int = 3):
        pred = self.model.predict(image.reshape(-1, self.image_size, self.image_size, 3))
        top_labels = {}
        if len(self.labels) >= n_top:
            top_labels_ids = np.flip(np.argsort(pred, axis=1)[0, -n_top:])
            for label_id in top_labels_ids:
                top_labels[self.labels[label_id]] = pred[0,label_id].item()
        pred_label = self.labels[np.argmax(pred)]
        print(top_labels)
        return {'label': pred_label, 'top': top_labels}

    def predict_file(self, file, n_top=3):
        img = np.array(Image.open(file).resize((self.image_size,self.image_size)), dtype=np.float32)
        return self.predict_image(img, n_top)