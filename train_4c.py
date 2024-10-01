from ultralytics import YOLO
import torch

def adjust_in_channel(model, in_channel):
    first_conv_layer = model.model.model[0].conv
    new_conv_layer = torch.nn.Conv2d(4, first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size, stride=first_conv_layer.stride, padding=first_conv_layer.padding, bias=first_conv_layer.bias is not None)
    with torch.no_grad():
        new_conv_layer.weight[:, :3, :, :] = first_conv_layer.weight  # keep 1~3 channels
        new_conv_layer.weight[:, 3:, :, :] = first_conv_layer.weight.mean(dim=1, keepdim=True)  # init fourth channel
    model.model.model[0].conv = new_conv_layer
    model.model.yaml['ch'] = 4  # modify cfg
    return model

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
    
    model = adjust_in_channel(model, 4)

    # Use the model
    model.train(data="datasets/test_4c/data.yaml", epochs=50, batch=8)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set