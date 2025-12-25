# Child Mind Institute - Problematic Internet Use

This project contains code for the "Child Mind Institute - Problematic Internet Use" data mining task. It involves analyzing data to predict problematice internet use based on physical activity and fitness data.

## Installation

### Prerequisites

- Python 3.12 or higher
- Conda (optional, for environment management)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd child-mind-institute-problematic-internet-use
   ```

2. **Download Data:**
   The `data/` directory is not included in the repository. Please download the dataset from Kaggle:
   
   [Child Mind Institute - Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)
   
   Extract the contents and place them into the `data/` folder in the root of the project.

3. **Create and activate a virtual environment:**
   
   *Using `venv` (standard Python):*
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
   
   *Using `conda`:*
   ```bash
   conda create -n cmi_piu python=3.12
   conda activate cmi_piu
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Models
(Adjust specific scripts based on your workflow)
```bash
python models/train_level1_base.py
python models/train_level2_meta.py
```

## Structure
- `data/`: Contains dataset files.
- `models/`: Training and inference scripts.
- `analysis/`: Feature engineering and analysis notebooks/scripts.
- `parsers/`: Data preprocessing tools.
