# ARGO Data Downloader & Training System

This system automatically downloads ARGO NetCDF files and trains PyTorch models for temperature-salinity prediction.

## 🌊 Features

- **Auto-downloads** ARGO NetCDF files from NOAA NCEI
- **Folder structure** preservation: `raw_data/argo/indian/<year>/<month>/`
- **Duplicate checking** - skips already downloaded files
- **Permission system** - asks for permission every 1GB downloaded
- **PyTorch training** - LSTM and 1D CNN models
- **Progress logging** - comprehensive logging and progress tracking
- **Memory optimized** - designed for 16GB M4 MacBook

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python main.py
```

### 3. Run Individual Phases
```bash
# Download only
python main.py --download-only

# Training only (requires existing data)
python main.py --train-only

# Skip dependency check
python main.py --skip-deps
```

## 📁 File Structure

```
FloatChat/
├── main.py                 # Main entrypoint
├── argo_downloader.py     # NetCDF downloader
├── argo_trainer.py        # PyTorch training
├── raw_data/              # Downloaded NetCDF files
│   └── argo/
│       └── indian/
│           ├── 2019/
│           │   ├── 01/
│           │   │   ├── *.nc
│           │   │   └── ...
│           │   └── 02/
│           └── 2020/
├── models/                # Trained models
│   ├── argo_model.pt      # Final model
│   ├── argo_model_lstm.pt # LSTM model
│   └── argo_model_cnn.pt  # CNN model
└── logs/                  # Log files
    ├── argo_download.log
    ├── argo_training.log
    └── argo_pipeline.log
```

## 🔧 Components

### ARGO Downloader (`argo_downloader.py`)
- **Web scraping** with BeautifulSoup
- **Folder discovery** - finds all year/month combinations
- **Duplicate checking** - skips existing files
- **Permission system** - asks every 1GB
- **Progress tracking** - detailed logging

### ARGO Trainer (`argo_trainer.py`)
- **Data loading** with xarray
- **Preprocessing** - normalization, NaN removal
- **PyTorch models** - LSTM and 1D CNN
- **Training loop** - with early stopping
- **Model saving** - with scalers and metadata

### Main Pipeline (`main.py`)
- **Dependency checking**
- **Phase management** - download → training
- **Error handling** - graceful failures
- **Progress reporting** - comprehensive summaries

## 🧠 Models

### LSTM Model
- **Architecture**: 2-layer LSTM with dropout
- **Input**: Temperature sequences
- **Output**: Salinity predictions
- **Hidden size**: 64 units
- **Dropout**: 0.2

### 1D CNN Model
- **Architecture**: 3-layer 1D CNN with max pooling
- **Input**: Temperature sequences
- **Output**: Salinity predictions
- **Filters**: 32, 64, 128
- **Kernel size**: 3

## 📊 Data Processing

1. **Load NetCDF files** with xarray
2. **Extract variables** - temperature and salinity
3. **Remove NaNs** - clean data
4. **Normalize** - StandardScaler
5. **Create sequences** - sliding window
6. **Split data** - 80% train, 20% validation

## 🔍 Logging

All operations are logged to:
- **Console** - real-time progress
- **Files** - detailed logs for debugging
- **Progress bars** - visual progress tracking

## ⚙️ Configuration

### Download Settings
- **Base URL**: `https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/`
- **Permission threshold**: 1GB
- **Timeout**: 30 seconds per request
- **Chunk size**: 8KB

### Training Settings
- **Epochs**: 50 (with early stopping)
- **Batch size**: 32
- **Learning rate**: 0.001
- **Sequence length**: 100 points
- **Patience**: 10 epochs

## 🛠️ Troubleshooting

### Common Issues

1. **Memory errors**: Reduce batch size or sequence length
2. **Download failures**: Check internet connection
3. **Training errors**: Ensure data is properly loaded
4. **Permission denied**: Check file permissions

### Debug Mode
```bash
# Enable debug logging
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python main.py
```

## 📈 Performance

### Memory Usage
- **Download**: ~100MB RAM
- **Training**: ~2-4GB RAM (depending on data size)
- **Storage**: ~1-10GB (depending on data downloaded)

### Speed
- **Download**: ~1-5 MB/s (depending on connection)
- **Training**: ~1-2 minutes per epoch
- **Total time**: 2-4 hours (depending on data size)

## 🔒 Safety Features

- **Duplicate checking** - won't re-download existing files
- **Permission system** - asks before large downloads
- **Error handling** - graceful failure recovery
- **Progress saving** - resumes from where it left off
- **Memory monitoring** - tracks RAM usage

## 📝 Example Usage

```python
# Download data
from argo_downloader import ARGODownloader
downloader = ARGODownloader()
downloader.download_all()

# Train models
from argo_trainer import ARGOTrainer
trainer = ARGOTrainer()
trainer.load_nc_files()
trainer.train_model("lstm", epochs=50)
```

## 🎯 Next Steps

1. **Run the pipeline**: `python main.py`
2. **Monitor progress**: Check log files
3. **Use trained models**: Load with PyTorch
4. **Integrate with dashboard**: Use models in FloatChatAI

---

**Note**: This system is optimized for 16GB M4 MacBook with 150GB free storage. Adjust parameters as needed for your system.


