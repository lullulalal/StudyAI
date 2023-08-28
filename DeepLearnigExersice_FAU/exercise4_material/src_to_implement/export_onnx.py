import torch as t
from trainer import Trainer
import sys
import model
import torchvision as tv

epoch = int(sys.argv[1])
#TODO: Enter your model here

crit = t.nn.BCELoss()
trainer = Trainer(model.ResNet(), crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
