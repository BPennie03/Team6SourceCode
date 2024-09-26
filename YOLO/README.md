# How to use YOLOv8

## Training/Testing a model
1. Install Ultralytics (YOLO): `pip install -r requirements.txt`
2. Train the model: `python3 train.py`
3. Add any images you want to test into the `test_images` directory
4. Run `python3 detect.py`
5. Verify output files: `YOLO/runs/detect/predict/`

### Fixing pathing errors
If you get this error:
```
Dataset 'data.yaml' images not found , missing path 'insert/path/here'
Note dataset download directory is 'insert/path/here'. You can update this in 'insert/path/here\Ultralytics\settings.json'
```

Simply update the paths inside the specified `settings.json` to reflect the proper project locations:

*example: `vim /absolute/path/to/Ultralytics/settings.json`*
```bash
"datasets_dir": "/absolute/path/to/Team6SourceCode/YOLO",
"weights_dir": "/absolute/path/to/Team6SourceCode/YOLO/weights",
"runs_dir": "/absolute/path/to/Team6SourceCode/YOLO/runs",
```

## Modifying YOLO training parameters
* `data` - Path to the dataset configuration file: `data.yaml` (should not be changed)
* `epochs` - Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance. (I do 20-30)
* `imgsz` - size of image, larger sizes can be more accurate but take **WAY** longer
* `device` - Specifies the computational device(s) for training: a single GPU `(device=0)`, multiple GPUs `(device=0,1)`, CPU `(device=cpu)`, or MPS for Apple silicon `(device=mps)`