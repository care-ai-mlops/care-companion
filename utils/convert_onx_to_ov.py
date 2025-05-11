import openvino as ov
ov_model = ov.convert_model("/home/satya/care-companion/models/onnx/chest/chest_resnet.onnx")
ov.save_model(ov_model, "/home/satya/care-companion/models/openvino/chest_resnet.xml")