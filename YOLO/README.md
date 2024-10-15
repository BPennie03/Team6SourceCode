# How to use YOLOv8
[Documentation](https://docs.ultralytics.com/)

## Usage
### train.py
* `python3 train.py`

### detect.py
* `python3 detect.py`
* `python3 detect.py -f {file_path}`

## Training/Testing a model
1. Install Ultralytics (YOLO): 
   - `pip install Ultralytics`
   - or simply `pip install -r requirements.txt` to install all project dependencies
2. Train the model: `python3 train.py` 
   - *This is required for each person as output files/models are fairly large and not included on git*
   - Update params as desired/needed. *See section at bottom of page.*
   - If you train multiple times, you will have multiple output models in the `runs/detect/` directory: "train", "train2", etc. Don't worry about adjusting pathing if that is the case, the script should automatically train with the most recent created model
3. Add any images you want to test into the `resources` directory (*if desired*)
4. Run `python3 detect.py` to run detection **after** model is trained
   - `detect.py` defaults to performing detection on every image in the `test_images` directory. 
   - An optional argument, `-f`, allows you to path to a specific file, which is helpful for testing a single image/file or an alternate directory. 
   - *example: `python3 detect.py -f {file_path}`*
5. View output files: `output/detect_results/`

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
* `epochs` - Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance. (I have found best results are >= 30)
* `imgsz` - size of image, larger sizes can be more accurate but take **WAY** longer. I highly recommend leaving this at '640'.
* `device` - Specifies the computational device(s) for training: a single GPU `(device=0)`, multiple GPUs `(device=0,1)`, CPU `(device=cpu)`, or MPS for Apple silicon `(device=mps)`
