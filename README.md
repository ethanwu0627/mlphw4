# Testing Instructions

## Testing LeNet1
To test the original LeNet5 model, modify the `test_dataloader` in the `main()` function of `test1.py` to load your grading dataset, then run:
```bash
python test1.py
```

## Testing LeNet2
To test the modified LeNet5 model, modify the `test_dataloader` in the `main()` function of `test2.py` to load your grading dataset, then run:
```bash
python test2.py
```

## Dataset Setup
Both test scripts expect a dataset that returns `(image, label)` tuples where:
- `image`: Preprocessed tensor (1×32×32) ready for model input
- `label`: Integer class label (0-9)

The current implementation uses the `MNISTParquet` class to load data from Hugging Face. To use your transformed test dataset, replace the dataset initialization in both test files with your custom dataset loader that follows the same `(image, label)` format.