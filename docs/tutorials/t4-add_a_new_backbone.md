# Add a new backbone

Code for this section：
```
core/model/backbone/*
config/backbones/*
```

If you want to add a new backbone in `LibFewShot`, you should put all files about this new backbone in directory `core/model/backbone/`. For example, to add a `ResNet` to `LibFewShot`, you need provide a `resnet.py` in directory `core/model/backbone/`, and provide a class or function that can return a ResNet model like following:

```python
...

class ResNet(nn.Module):
	def __init(self,...):
...

def ResNet18():
	model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
	return model
```

After that, to make sure `trainer.py` could call `ResNet18`, you need add a line in `core/model/backbone/__init__.py` as follows:

```python
...

from resnet import ResNet18
```

At this point, the addition of a new backbone is finished.

The new backbone shares the same way to use as previous backbone. For example, to change `DN4` backbone to the new backbone, you just modify `backbone`'s value in ' `config/dn4.yaml` as follows:

```yaml
# arch info
backbone:
  name: resnet18
  kwargs:
    avg_pool: False
    is_flatten: False
```
