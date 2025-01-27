import torch

def input_shape_split_into_lidar_and_sensor(input_shape: {str: (int, ...)}) -> ({str: (int, ...)}, {str: (int, ...)}):
    lidar = {}
    sensor = {}

    for name, shape in input_shape.items():
        if "lidar" in name:
            lidar[name] = shape
        else:
            sensor[name] = shape

    return lidar, sensor

def split_input_shape(input_shape: {str: (int, ...)}) -> ({str: (int, ...)}, {str: (int, ...)}, {str: (int, ...)}):
    lidar = {}
    sensor = {}
    goal = {}

    for name, shape in input_shape.items():
        if "lidar" in name:
            lidar[name] = shape
        elif "sensor" in name:
            sensor[name] = shape
        else:
            goal[name] = shape

    return lidar, sensor, goal


def observation_split_into_lidar_and_sensor(observation: {str, torch.Tensor}) -> ({str, torch.Tensor}, {str, torch.Tensor}):
    lidar = {}
    sensor = {}
    for name, tensor in observation.items():
        if "lidar" in name:
            lidar[name] = tensor
        else:
            sensor[name] = tensor
    return lidar, sensor

def split_observation(observation: {str, torch.Tensor}) -> ({str, torch.Tensor}, {str, torch.Tensor}, {str, torch.Tensor}):
    lidar = {}
    sensor = {}
    goal = {}
    for name, tensor in observation.items():
        if "lidar" in name:
            lidar[name] = tensor
        elif "sensor" in name:
            sensor[name] = tensor
        else:
            goal[name] = tensor
    return lidar, sensor, goal

def input_shape_calculate_flatten_size(input_shape: {str: (int, ...)}) -> int:
    size = 0

    for name, shape in input_shape.items():
        if len(shape) != 0:
            result = 1
            for value in shape:
                result *= value
            size += result

    return size


def output_shape_calculate_from_model_flatten(input_shape: (int, ...), model: torch.nn.Module) -> int:
    return model(torch.zeros(1, *input_shape)).flatten(1).data.size(1)
