class Encoder():
    def __init__(self, num_landmarks, num_dimensions):
        self.num_landmarks = num_landmarks
        self.num_dimensions = num_dimensions

    def get_parameters_length(self):
        return self.num_landmarks * self.num_dimensions

    def encode_parameters(self, decoded):
        return decoded.reshape((decoded.shape[0], self.num_landmarks * self.num_dimensions))

    def decode_parameters(self, encoded):
        return encoded.reshape((encoded.shape[0], self.num_landmarks, self.num_dimensions))

    def transcode_parameters(self, decoded):
        return decoded

    def encode_deltas(self, shapes_prev, shapes_curr):
        return self.encode_parameters(shapes_curr) - self.encode_parameters(shapes_prev)

    def decode_deltas(self, params_old, params_delta):
        return self.decode_parameters(params_old + params_delta)
