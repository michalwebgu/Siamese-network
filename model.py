from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Concatenate, Convolution2D, MaxPooling2D

class siamese_model:
    def __init__(self, input_shape, num_classes, embedding_size):
            # Convolutional Neural Network
            self.model = Sequential(name="conv_net")
            self.model.add(Convolution2D(32, 3, 3, input_shape=input_shape, activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            # 32 filters each. The kernel size is 3*3
            self.model.add(Convolution2D(32, 3, 3, activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            # fully connected part
            self.model.add(Flatten())
            self.model.add(Dense(embedding_size, activation='relu', name='embedding'))
            self.model.add(Dense(num_classes, activation='softmax'))

            # Define the tensors for the two input images
            self.x1 = Input(input_shape, name="x1")
            self.x2 = Input(input_shape, name="x2")

    
            # to get the output of the embedding layer, create a sub model from input layer to embedding layer
            embedding_sub_model = Model(inputs=self.model.input,
                                             outputs=self.model.get_layer('embedding').output)
                                             
            self.embedding_x1 = embedding_sub_model(self.x1)
            self.embedding_x2 = embedding_sub_model(self.x2)

            # Generate model output
            self.model_output_x1 = self.model(self.x1)
            self.model_output_x2 = self.model(self.x2)

            # Concatenate outputs (so it can be used in the same loss function)
            concatted = Concatenate()([self.model_output_x1, self.model_output_x2,
                                       self.embedding_x1, self.embedding_x2])

            # Connect the inputs with the outputs
            self.siamese = Model(inputs=[self.x1, self.x2],
                                 outputs=[concatted])
