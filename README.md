🎬 Smart Ad Placement

Predict the best moments to insert ads in movies & videos using deep learning
What is this?
This is a deep learning project that watches videos (their subtitles, actually) and finds the best places to put ads. Instead of randomly interrupting the video, it learns to find natural breaks - like scene changes, quiet moments, or emotional shifts - where ads won't annoy viewers.
We built this because we all hate when ads pop up at the worst possible moment (like during a dialogue or action scene). This AI tries to be smarter about that.
Why this matters?
Streaming platforms need ads to make money, but viewers hate bad ad timing. When ads come at the wrong time:

* People get frustrated and close the app

* They skip ads or use ad-blockers

* Platform loses revenue

Good ad placement = happy viewers + more ad views = everyone wins.
How it works (the simple version)

1. Read subtitles - The AI reads movie subtitles to understand what's happening

2. Find patterns - It learns patterns: gaps between speech, music moments, scene changes

3. Score moments - Each moment gets a score: "how good is this for an ad?"

4. Select best spots - Pick the highest-scoring moments that aren't too close together

The model looks at both text (what characters say) and timing (when they say it) to make decisions.
What I built

* BiLSTM Neural Network - Processes subtitle text forward and backward to understand context

* Feature Engineering - Creates smart features like gaps, speaking rate, punctuation

* Smart Selection Algorithm - Picks ad spots with proper spacing (no ad clusters)

* Streamlit Web App - Upload subtitles, get ad suggestions instantly

* Class Imbalance Handling - Uses SMOTE + Focal Loss (77:1 imbalance is tough!)

* Threshold Optimization - Find the right balance between precision and recall

Tech Stuff

* Languages: Python 3.13+

* Deep Learning: TensorFlow 2.15 + Keras

* Model: Dual-Branch BiLSTM (Text + Numeric features)

* Preprocessing: Pandas, NumPy, Regex

* App: Streamlit

* ML Tools: Scikit-learn, imbalanced-learn

* Data: Subtitle files (.srt, .txt, .csv)

Let's get started
1. Clone the repo

```
git clone [https://github.com/your-username/ad-placement-ai.git](https://github.com/your-username/ad-placement-ai.git) 
cd ad-placement-ai
```

2. Create virtual environment (do this!)

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

Make sure you have these packages:

```
streamlit==1.29.0
tensorflow==2.15.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
imbalanced-learn==0.12.2
matplotlib==3.8.2
```

4. Download model files
You need these files in your project root:

* bilstm_ad_model.weights.h5 (trained model weights)

* tokenizer.pkl (text tokenizer)

Download from Releases or train your own (see below).
5. Run the app

```
streamlit run app.py
```

The app opens in your browser automatically.
How to use the app

* Upload subtitle file (.srt, .txt, or .csv)

* Adjust settings in sidebar:

  * Decision threshold (lower = more ads)

  * Minimum spacing between ads

  * Skip intro/end credits

* Click "Run Detection"

* Get results:

  * Table with ad timestamps

  * Visual timeline showing ad positions

  * Download CSV for later use

Tip: Start with threshold 0.50, then adjust based on results.
How to train your own model
If you want to train from scratch:

```
python train.py \
  --dataset /path/to/subtitles_labeled.parquet \
  --output ./models \
  --epochs 5 \
  --batch-size 512
```

The training script:

* Loads labeled subtitle data (with "is_ad" labels)

* Splits data maintaining class distribution

* Applies SMOTE to balance training set

* Trains BiLSTM with Focal Loss

* Saves model, tokenizer, and training history

* Evaluates on test set

Training data format
Your dataset should be a Parquet file with these columns:

* text: Subtitle text (string)

* norm_gap: Normalized gap time (float 0-1)

* norm_duration: Normalized duration (float 0-1)

* is_sentence_end: 1 if ends with punctuation (int)

* has_music_tag: 1 if contains music/sound (int)

* is_shouting: 1 if text is uppercase/exclamation (int)

* is_ad: 1 if this is a good ad placement (int) - your label



```
Input Text → Tokenizer → Embedding (128d) → 
BiLSTM (128 units) → BiLSTM (64 units) → 
Dropout → Dense (128) → Output
```

Numeric Branch (Dense)

```
Numeric Features → Dense (64) → Dropout → 
Dense (32) → Dropout → Output
```

Combined

```
[Text Output] + [Numeric Output] → Concatenate → 
Dense (128) → Dropout → Dense (64) → 
Dropout → Sigmoid (final probability)
```

Total params: ~1.48M (5.66 MB)
Key Features
Feature
What it means
Why it matters
norm_gap
Time gap after subtitle
Long gap = good for ad
is_sentence_end
Ends with .!?
Natural break point
has_music_tag
Music/sound effect
Good ad moment
is_shouting
Yelling/excitement
Bad ad moment (interrupts action)
Project Structure

```
ad-placement-ai/
├── app.py                    # Streamlit web app
├── train.py                  # Training script
├── model.py                  # Model architecture
├── preprocess.py             # Data cleaning & feature engineering
├── requirements.txt          # Python dependencies
├── bilstm_ad_model.weights.h5 # Model weights (download separately)
├── tokenizer.pkl             # Tokenizer (download separately)
├── data/
│   ├── raw/                  # Raw subtitle files
│   ├── processed/            # Cleaned features
│   └── subtitles_labeled.parquet # Training data
├── notebooks/
│   ├── eda.ipynb             # Exploratory analysis
│   └── model_experiments.ipynb # Model testing
└── tests/                    # Unit tests
```

Results (What I got)
After training on 647K subtitle samples:
Metric
Value
What it means
Test Accuracy
88.26%
Overall correct predictions
Test AUC
74.77%
Model discrimination ability
Precision (Ad class)
5.1%
Only 5% of predicted ads are real ads 
Classes
1.28% ads
Extreme imbalance (77:1)
The Problem
Even with SMOTE and Focal Loss, precision is terrible because:

* 98.72% of data is "no-ad" class

* Model learns to predict "no-ad" most of the time

* When it does predict "ad", it's usually wrong

How We fixed it

* Threshold tuning: Lower threshold = more ad predictions

* Post-processing: Smart selection algorithm filters bad predictions

* Hybrid approach: Use model + heuristic rules together
