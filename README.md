---
title: GLiNER Hotel NER Demo
emoji: 🏨
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.1
app_file: src/streamlit_app.py
pinned: false
license: mit
---

# GLiNER Hotel Named Entity Recognition Demo

A powerful demonstration of the **GLiNER** (Generalist and Lightweight Named Entity Recognition) model, specialized for hotel entity recognition tasks.

## 🚀 Features

- **🔍 Interactive Text Analysis**: Extract entities from any text using custom labels
- **📊 Hotel Dataset Evaluation**: Evaluate model performance on hotel descriptions
- **🎯 Real-time Processing**: Fast entity recognition with confidence scoring
- **📈 Performance Metrics**: Precision, Recall, and F1-score calculations
- **🎨 Visual Highlighting**: Color-coded entity highlighting in text

## 🏨 Hotel Dataset

The demo includes a curated dataset of hotel descriptions with ground truth annotations for hotel name recognition. The dataset contains:

- **4 hotel descriptions** from various properties
- **Ground truth annotations** for hotel names
- **Real booking.com style content** with rich details

## 🤖 Model Information

This demo uses the [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) model, which is a state-of-the-art generalist NER model that can identify entities based on custom labels without requiring fine-tuning.

## 📊 Evaluation Modes

### Single Text Analysis
- Input custom text and entity labels
- Adjustable confidence threshold
- Interactive entity extraction
- Visual highlighting and statistics

### Hotel Dataset Evaluation
- Benchmark model performance on hotel data
- Sample-by-sample analysis
- Cached predictions for fast sample switching
- Detailed comparison of predictions vs. ground truth

## 🔧 Technical Details

- **Framework**: Streamlit for interactive web interface
- **Model**: GLiNER multi-v2.1 for entity recognition
- **Caching**: Smart caching for model loading and evaluations
- **Performance**: Optimized for fast inference and user experience

## 📝 Usage

1. **Select Mode**: Choose between single text analysis or dataset evaluation
2. **Configure Settings**: Adjust confidence threshold and entity labels
3. **Run Analysis**: Click the processing button to extract entities
4. **Explore Results**: View highlighted text, statistics, and detailed metrics

## 🎯 Use Cases

Perfect for:
- **Hotel Industry**: Extract hotel names from descriptions
- **Travel Platforms**: Process booking content
- **Content Analysis**: Analyze hospitality texts
- **Research**: Benchmark NER model performance
- **Education**: Learn about named entity recognition

## 🔗 Links

- **Model**: [GLiNER on Hugging Face](https://huggingface.co/urchade/gliner_multi-v2.1)
- **Framework**: [GLiNER GitHub](https://github.com/urchade/GLiNER)
- **Interface**: [Streamlit](https://streamlit.io/)

---

Built with ❤️ using Streamlit and GLiNER