from tensorflow.keras.utils import Sequence
import random
import numpy as np
import cv2
import augmentation

def get_image(image_path, input_shape):
    # read image from file as a BGR array
    img = cv2.imread(image_path)
    # convert image to RGB
    img = img[..., ::-1]
    # rescale values
    img = img / 255

    img = cv2.resize(img, (input_shape[1], input_shape[0]) , interpolation = cv2.INTER_AREA)

    return img

class DataGenerator(Sequence):
    def __init__(self, data, input_shape, embedding_size,
                 steps_per_epoch, images_dir, augmentations = None, batch_size=16):
        self.data = data
        self.steps_per_epoch = steps_per_epoch
        self.images_dir = images_dir
        self.augmentations = augmentations
        self.input_shape = (input_shape[0], input_shape[1])
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        #create dict to hold brand dfs
        self.brands_df_dict = {}

        for brand in list(data.brand.unique()):
          self.brands_df_dict[brand] = data[data.brand == brand]

        self.brands = list(self.brands_df_dict.keys())

        # create class to index mapping
        self.class_to_index = {}
        for brand_name in self.brands:
            self.class_to_index[brand_name] = self.brands.index(brand_name) + 1
        self.num_classes = len(list(self.class_to_index.keys()))

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        x1_batch = []
        x2_batch = []
        y1_batch = []
        y2_batch = []
        # batches to hold dummy y_true embeddings
        y3_dummie_batch = []
        y4_dummie_batch = []

        # same brand
        n_samples = int(self.batch_size/2)
        for i in range(n_samples):
            brand_name = random.choice(self.brands)
            brand_rows = self.brands_df_dict[brand_name].sample(n=2)
            x1_image_path = self.images_dir + '/' + brand_rows.iloc[0].filename
            x2_image_path = self.images_dir + '/' + brand_rows.iloc[1].filename
            x1 = get_image(x1_image_path, self.input_shape)
            x2 = get_image(x2_image_path, self.input_shape)
            if(self.augmentations):
                x1 = self.augmentations(x1)
                x2 = self.augmentations(x2)
            y1 = y2 = np.eye(self.num_classes)[self.class_to_index[brand_name]]
            x1_batch.append(x1)
            x2_batch.append(x2)
            y1_batch.append(y1)
            y2_batch.append(y2)
            y3_dummie_batch.append([0 for i in (range(self.embedding_size))])
            y4_dummie_batch.append([0 for i in (range(self.embedding_size))])


        # different brands
        n_samples = int(self.batch_size / 2)
        for i in range(n_samples):
            brand_name_1 = random.choice(self.brands)
            brand_name_2 = random.choice(self.brands)
            while(brand_name_1 == brand_name_2):
                brand_name_2 = random.choice(self.brands)
            brand_1_row = self.brands_df_dict[brand_name_1].sample(n=1)
            brand_2_row = self.brands_df_dict[brand_name_2].sample(n=1)
            x1_image_path = self.images_dir + '/' + brand_1_row.iloc[0].filename
            x2_image_path = self.images_dir + '/' + brand_2_row.iloc[0].filename
            x1 = get_image(x1_image_path, self.input_shape)
            x2 = get_image(x2_image_path, self.input_shape)
            if (self.augmentations):
                x1 = self.augmentations(x1)
                x2 = self.augmentations(x2)
            y1 = np.eye(self.num_classes)[self.class_to_index[brand_name_1]]
            y2 = np.eye(self.num_classes)[self.class_to_index[brand_name_2]]
            x1_batch.append(x1)
            x2_batch.append(x2)
            y1_batch.append(y1)
            y2_batch.append(y2)
            y3_dummie_batch.append([0 for i in (range(self.embedding_size))])
            y4_dummie_batch.append([0 for i in (range(self.embedding_size))])

       
        x1_batch = np.array(x1_batch)
        x2_batch = np.array(x2_batch)
        y1_batch = np.array(y1_batch)
        y2_batch = np.array(y2_batch)
        y3_dummie_batch = np.array(y3_dummie_batch)
        y4_dummie_batch = np.array(y4_dummie_batch)

        x_batch = [x1_batch, x2_batch]
        y_batch = [y1_batch, y2_batch, y3_dummie_batch, y4_dummie_batch]

        # Concatenate outputs (so it can be used in the same loss function)
        y_batch = np.concatenate(y_batch, axis=1)

        return x_batch, y_batch
